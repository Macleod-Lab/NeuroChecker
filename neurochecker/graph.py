from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label as nd_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from neurochecker.mask_io import MaskEntry, load_mask_array


@dataclass
class Node:
    id: int
    frame: int
    run_id: str
    x_px: float
    y_px: float
    z_frame: int
    x: float
    y: float
    z: float
    mask_path: str
    mask_x: int
    mask_y: int
    mask_width: int
    mask_height: int
    color: Optional[Tuple[int, int, int]] = None
    degree: int = 0
    label: str = "normal"


@dataclass
class GraphResult:
    nodes: List[Node]
    edges: List[Tuple[int, int]]
    order: List[int]
    paths: List[List[int]]
    counts: Dict[str, int]


def _compute_centroid(mask: np.ndarray, x0: int, y0: int) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return (x0 + float(xs.mean()), y0 + float(ys.mean()))


def _component_centroids(mask: np.ndarray, x0: int, y0: int, *, min_area_px: int = 25) -> List[Tuple[float, float]]:
    labeled, num = nd_label(mask > 0)
    if num <= 0:
        return []
    centroids: List[Tuple[float, float]] = []
    for idx in range(1, num + 1):
        ys, xs = np.nonzero(labeled == idx)
        if ys.size == 0:
            continue
        if ys.size < min_area_px:
            continue
        centroids.append((x0 + float(xs.mean()), y0 + float(ys.mean())))
    return centroids


def build_nodes(
    entries: Sequence[MaskEntry],
    *,
    pixel_size_xy: float,
    slice_thickness_z: float,
    include_runs: Optional[Sequence[str]] = None,
) -> List[Node]:
    nodes: List[Node] = []
    include_set = set(r for r in include_runs) if include_runs else None
    for entry in entries:
        if include_set is not None and entry.run_id not in include_set:
            continue
        mask = load_mask_array(entry)
        if mask is None:
            continue
        h, w = mask.shape[:2]
        if entry.width <= 0:
            entry.width = w
        if entry.height <= 0:
            entry.height = h
        centroids = _component_centroids(mask, entry.x, entry.y)
        if not centroids:
            continue
        z = int(entry.frame)
        for cx, cy in centroids:
            nodes.append(
                Node(
                    id=len(nodes),
                    frame=z,
                    run_id=entry.run_id,
                    x_px=cx,
                    y_px=cy,
                    z_frame=z,
                    x=cx * pixel_size_xy,
                    y=cy * pixel_size_xy,
                    z=z * slice_thickness_z,
                    mask_path=str(entry.path),
                    mask_x=entry.x,
                    mask_y=entry.y,
                    mask_width=entry.width,
                    mask_height=entry.height,
                    color=entry.color,
                )
            )
    return nodes


def _build_knn_edges(
    coords: np.ndarray,
    nodes: Sequence[Node],
    *,
    k_neighbors: int,
    max_frame_gap: Optional[int],
    max_distance: Optional[float],
) -> List[Tuple[int, int, float]]:
    if coords.shape[0] == 0:
        return []
    k = max(1, min(k_neighbors + 1, coords.shape[0]))
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k)
    edges: List[Tuple[int, int, float]] = []
    for i, (row_dists, row_idxs) in enumerate(zip(dists, idxs)):
        for dist, j in zip(row_dists[1:], row_idxs[1:]):
            if i == j:
                continue
            if max_frame_gap is not None:
                if abs(nodes[i].frame - nodes[j].frame) > max_frame_gap:
                    continue
            if max_distance is not None and dist > max_distance:
                continue
            edges.append((i, int(j), float(dist)))
    return edges


def _build_frame_link_edges(
    coords: np.ndarray,
    nodes: Sequence[Node],
    *,
    max_distance: Optional[float],
) -> List[Tuple[int, int, float]]:
    if coords.shape[0] == 0:
        return []
    by_frame: Dict[int, List[int]] = {}
    for idx, node in enumerate(nodes):
        by_frame.setdefault(node.frame, []).append(idx)
    edges: List[Tuple[int, int, float]] = []
    frames = sorted(by_frame.keys())
    if len(frames) < 2:
        return edges
    for frame in frames[1:]:
        prev_frame = frame - 1
        if prev_frame not in by_frame:
            continue
        prev_ids = by_frame[prev_frame]
        curr_ids = by_frame[frame]
        prev_coords = coords[prev_ids]
        tree = cKDTree(prev_coords)
        for curr_id in curr_ids:
            dist, local_idx = tree.query(coords[curr_id], k=1)
            if max_distance is not None and dist > max_distance:
                continue
            prev_id = prev_ids[int(local_idx)]
            edges.append((curr_id, prev_id, float(dist)))
    return edges


def build_graph(
    nodes: Sequence[Node],
    *,
    mode: str = "mst",
    k_neighbors: int = 6,
    max_frame_gap: Optional[int] = 3,
    max_distance: Optional[float] = None,
) -> GraphResult:
    if not nodes:
        return GraphResult(nodes=list(nodes), edges=[], order=[], paths=[], counts={"nodes": 0, "edges": 0})
    coords = np.asarray([[n.x, n.y, n.z] for n in nodes], dtype=np.float64)
    if mode == "frame_link":
        edge_candidates = _build_frame_link_edges(coords, nodes, max_distance=max_distance)
        edges = [(i, j) for i, j, _ in edge_candidates]
    else:
        k_neighbors = max(k_neighbors, 2)
        edge_candidates = _build_knn_edges(
            coords,
            nodes,
            k_neighbors=k_neighbors,
            max_frame_gap=max_frame_gap,
            max_distance=max_distance,
        )
        if not edge_candidates:
            edges = []
        else:
            rows = []
            cols = []
            data = []
            for i, j, dist in edge_candidates:
                rows.append(i)
                cols.append(j)
                data.append(dist)
                rows.append(j)
                cols.append(i)
                data.append(dist)
            graph = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
            mst = minimum_spanning_tree(graph)
            mst = mst.tocoo()
            edges = [(int(i), int(j)) for i, j in zip(mst.row, mst.col)]

    return label_graph(nodes, edges)


def label_graph(
    nodes: Sequence[Node],
    edges: Sequence[Tuple[int, int]],
    *,
    min_branch_length: int = 3,
) -> GraphResult:
    if not nodes:
        return GraphResult(nodes=list(nodes), edges=list(edges), order=[], paths=[], counts={"nodes": 0, "edges": 0})
    degrees = [0] * len(nodes)
    adjacency: Dict[int, List[int]] = {n.id: [] for n in nodes}
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
        adjacency[i].append(j)
        adjacency[j].append(i)

    def branch_length(prev: int, cur: int) -> int:
        length = 1
        while degrees[cur] == 2:
            next_nodes = [x for x in adjacency[cur] if x != prev]
            if not next_nodes:
                break
            prev, cur = cur, next_nodes[0]
            length += 1
        return length

    valid_branch_points: List[int] = []
    for idx, degree in enumerate(degrees):
        if degree < 3:
            continue
        lengths = []
        for nbr in adjacency[idx]:
            if degrees[nbr] >= 3:
                lengths.append(min_branch_length)
            else:
                lengths.append(branch_length(idx, nbr))
        if lengths and all(length >= min_branch_length for length in lengths):
            valid_branch_points.append(idx)

    valid_endpoints: List[int] = []
    for idx, degree in enumerate(degrees):
        if degree != 1:
            continue
        nbr = adjacency[idx][0]
        if branch_length(idx, nbr) >= min_branch_length:
            valid_endpoints.append(idx)

    endpoint_count = 0
    branch_count = 0
    isolated_count = 0
    valid_branch_set = set(valid_branch_points)
    valid_end_set = set(valid_endpoints)
    for node, degree in zip(nodes, degrees):
        node.degree = int(degree)
        if degree == 0:
            node.label = "isolated"
            isolated_count += 1
        elif node.id in valid_branch_set:
            node.label = "branch"
            branch_count += 1
        elif node.id in valid_end_set:
            node.label = "endpoint"
            endpoint_count += 1
        else:
            node.label = "normal"

    paths = _extract_paths(nodes, adjacency, degrees, valid_branch_set, valid_end_set)
    order = _build_traversal_order(nodes, list(edges), paths=paths)
    counts = {
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "endpoints": int(endpoint_count),
        "branchpoints": int(branch_count),
        "isolated": int(isolated_count),
    }
    return GraphResult(nodes=list(nodes), edges=list(edges), order=order, paths=paths, counts=counts)


def _extract_paths(
    nodes: Sequence[Node],
    adjacency: Dict[int, List[int]],
    degrees: Sequence[int],
    branch_set: set,
    end_set: set,
) -> List[List[int]]:
    if not nodes:
        return []
    visited_edges = set()
    paths: List[List[int]] = []

    def mark(u: int, v: int) -> None:
        visited_edges.add((u, v))
        visited_edges.add((v, u))

    def is_junction_or_end(node_id: int) -> bool:
        if degrees[node_id] == 0:
            return True
        if node_id in branch_set or node_id in end_set:
            return True
        return degrees[node_id] != 2

    for start in range(len(nodes)):
        if not is_junction_or_end(start):
            continue
        for nbr in adjacency[start]:
            if (start, nbr) in visited_edges:
                continue
            path = [start]
            prev = start
            cur = nbr
            mark(prev, cur)
            while degrees[cur] == 2:
                path.append(cur)
                next_nodes = [x for x in adjacency[cur] if x != prev]
                if not next_nodes:
                    break
                prev, cur = cur, next_nodes[0]
                mark(prev, cur)
            if path[-1] != cur:
                path.append(cur)
            paths.append(path)
    return paths


def _build_traversal_order(
    nodes: Sequence[Node],
    edges: Sequence[Tuple[int, int]],
    *,
    paths: Optional[Sequence[Sequence[int]]] = None,
) -> List[int]:
    if not nodes:
        return []
    order: List[int] = []

    def _root_key(node_id: int) -> Tuple[int, float, float]:
        node = nodes[node_id]
        return (node.frame, node.x_px, node.y_px)

    if paths:
        sorted_paths = sorted(paths, key=lambda p: _root_key(p[0]))
        seen = set()
        for path in sorted_paths:
            for node_id in path:
                if node_id in seen:
                    continue
                seen.add(node_id)
                order.append(node_id)
        for node_id in range(len(nodes)):
            if node_id not in seen:
                order.append(node_id)
        return order

    adjacency: Dict[int, List[int]] = {n.id: [] for n in nodes}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    remaining = set(adjacency.keys())

    while remaining:
        root = min(remaining, key=_root_key)
        stack = [root]
        while stack:
            current = stack.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            order.append(current)
            neighbors = sorted(adjacency[current], key=_root_key, reverse=True)
            stack.extend(neighbors)
    return order


def find_connected_components(
    nodes: Sequence[Node],
    edges: Sequence[Tuple[int, int]],
) -> List[List[int]]:
    """Return connected components as lists of node IDs, largest first."""
    adjacency: Dict[int, List[int]] = {n.id: [] for n in nodes}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    visited: set = set()
    components: List[List[int]] = []
    for node in nodes:
        if node.id in visited:
            continue
        component: List[int] = []
        stack = [node.id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            component.append(nid)
            stack.extend(adjacency[nid])
        components.append(component)
    components.sort(key=len, reverse=True)
    return components


def find_bridge_candidates(
    nodes: Sequence[Node],
    main_ids: Sequence[int],
    disconnected_ids: Sequence[int],
) -> Tuple[int, int, float, int]:
    """Find the closest pair of nodes between two components.

    Returns (disc_node_id, main_node_id, distance, frame_gap).
    """
    main_coords = np.asarray(
        [[nodes[nid].x, nodes[nid].y, nodes[nid].z] for nid in main_ids],
        dtype=np.float64,
    )
    disc_coords = np.asarray(
        [[nodes[nid].x, nodes[nid].y, nodes[nid].z] for nid in disconnected_ids],
        dtype=np.float64,
    )
    tree = cKDTree(main_coords)
    dists, indices = tree.query(disc_coords, k=1)
    best_idx = int(np.argmin(dists))
    disc_node = disconnected_ids[best_idx]
    main_node = main_ids[int(indices[best_idx])]
    distance = float(dists[best_idx])
    frame_gap = abs(nodes[disc_node].frame - nodes[main_node].frame)
    return disc_node, main_node, distance, frame_gap
