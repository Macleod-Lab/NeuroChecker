from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

from neurochecker.graph import GraphResult, Node, label_graph


@dataclass(frozen=True)
class HillockPruneResult:
    graph: GraphResult
    kept_original_node_ids: List[int]
    excluded_original_node_ids: List[int]
    original_to_current: Dict[int, int]
    current_to_original: List[int]
    hillock_original_node_id: int
    hillock_current_node_id: int
    distal_original_node_id: int
    distal_current_node_id: int
    forward_neighbor_original_node_id: int
    forward_neighbor_current_node_id: int


@dataclass(frozen=True)
class HillockSomaResult:
    graph: GraphResult
    current_to_original: List[int]
    original_to_current: Dict[int, int]
    segment_nodes: List[List[int]]
    segment_edges: Dict[int, Set[Tuple[int, int]]]
    soma_original_node_ids: List[int]
    soma_segment_index: int
    primary_neurite_segment_index: int
    hillock_original_node_id: int
    distal_original_node_id: int
    forward_neighbor_original_node_id: int


def clone_graph(graph: GraphResult) -> GraphResult:
    nodes = [replace(node, id=idx) for idx, node in enumerate(graph.nodes)]
    return GraphResult(
        nodes=nodes,
        edges=[(int(i), int(j)) for i, j in graph.edges],
        order=list(graph.order),
        paths=[list(path) for path in graph.paths],
        counts=dict(graph.counts),
    )


def _adjacency(node_count: int, edges: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = {idx: [] for idx in range(node_count)}
    for i, j in edges:
        if 0 <= i < node_count and 0 <= j < node_count:
            adjacency[i].append(j)
            adjacency[j].append(i)
    return adjacency


def _path_between(
    node_count: int,
    edges: Sequence[Tuple[int, int]],
    start_id: int,
    end_id: int,
) -> Optional[List[int]]:
    if not (0 <= start_id < node_count and 0 <= end_id < node_count):
        return None
    if start_id == end_id:
        return [start_id]
    adjacency = _adjacency(node_count, edges)
    queue = [start_id]
    parent: Dict[int, Optional[int]] = {start_id: None}
    index = 0
    while index < len(queue):
        current = queue[index]
        index += 1
        if current == end_id:
            break
        for neighbor in adjacency.get(current, []):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if end_id not in parent:
        return None
    path: List[int] = []
    current: Optional[int] = end_id
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path


def _sorted_node_ids(nodes: Sequence[Node], node_ids: Sequence[int]) -> List[int]:
    return sorted(
        [int(node_id) for node_id in node_ids],
        key=lambda node_id: (
            int(nodes[node_id].frame),
            float(nodes[node_id].x_px),
            float(nodes[node_id].y_px),
            int(node_id),
        ),
    )


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _component_from_seed_without_edge(
    adjacency: Dict[int, List[int]],
    *,
    seed: int,
    blocked_edge: Tuple[int, int],
) -> Set[int]:
    blocked = _edge_key(int(blocked_edge[0]), int(blocked_edge[1]))
    seen: Set[int] = set()
    stack = [int(seed)]
    while stack:
        current = int(stack.pop())
        if current in seen:
            continue
        seen.add(current)
        for neighbor in adjacency.get(current, []):
            if _edge_key(current, int(neighbor)) == blocked:
                continue
            if neighbor not in seen:
                stack.append(int(neighbor))
    return seen


def build_soma_aware_segments(
    graph: GraphResult,
    *,
    hillock_node_id: int,
    distal_node_id: int,
) -> HillockSomaResult:
    base_graph = clone_graph(graph)
    node_count = len(base_graph.nodes)
    if not (0 <= hillock_node_id < node_count):
        raise ValueError(f"Hillock node is out of range: {hillock_node_id}")
    if not (0 <= distal_node_id < node_count):
        raise ValueError(f"Distal node is out of range: {distal_node_id}")
    if hillock_node_id == distal_node_id:
        raise ValueError("Hillock node and distal node must be different.")

    path = _path_between(node_count, base_graph.edges, hillock_node_id, distal_node_id)
    if not path or len(path) < 2:
        raise ValueError("Could not find a neurite path from the hillock node to the distal node.")
    forward_neighbor = int(path[1])

    adjacency = _adjacency(node_count, base_graph.edges)
    soma_node_ids = _component_from_seed_without_edge(
        adjacency,
        seed=int(hillock_node_id),
        blocked_edge=(int(hillock_node_id), forward_neighbor),
    )
    soma_sorted = _sorted_node_ids(base_graph.nodes, soma_node_ids)

    soma_edge_set: Set[Tuple[int, int]] = set()
    for i, j in base_graph.edges:
        edge = _edge_key(int(i), int(j))
        if edge == _edge_key(int(hillock_node_id), forward_neighbor):
            continue
        if i in soma_node_ids and j in soma_node_ids:
            soma_edge_set.add(edge)

    segment_nodes: List[List[int]] = [list(soma_sorted)]
    segment_edges: Dict[int, Set[Tuple[int, int]]] = {0: set(soma_edge_set)}
    primary_neurite_segment_index = -1

    base_paths = [list(path) for path in base_graph.paths if path]
    if not base_paths:
        relabeled = label_graph(base_graph.nodes, base_graph.edges)
        base_paths = [list(path) for path in relabeled.paths if path]

    for path_nodes in base_paths:
        normalized = [int(node_id) for node_id in path_nodes]
        if not any(node_id not in soma_node_ids for node_id in normalized):
            continue
        seg_index = len(segment_nodes)
        segment_nodes.append(normalized)
        edge_set: Set[Tuple[int, int]] = set()
        for a, b in zip(normalized, normalized[1:]):
            if a == b:
                continue
            edge_set.add(_edge_key(int(a), int(b)))
        segment_edges[seg_index] = edge_set
        if primary_neurite_segment_index < 0 and forward_neighbor in normalized:
            primary_neurite_segment_index = seg_index

    if primary_neurite_segment_index < 0:
        fallback = [int(hillock_node_id), int(forward_neighbor)]
        primary_neurite_segment_index = len(segment_nodes)
        segment_nodes.append(fallback)
        segment_edges[primary_neurite_segment_index] = {
            _edge_key(int(hillock_node_id), int(forward_neighbor))
        }

    return HillockSomaResult(
        graph=base_graph,
        current_to_original=list(range(node_count)),
        original_to_current={idx: idx for idx in range(node_count)},
        segment_nodes=segment_nodes,
        segment_edges=segment_edges,
        soma_original_node_ids=soma_sorted,
        soma_segment_index=0,
        primary_neurite_segment_index=int(primary_neurite_segment_index),
        hillock_original_node_id=int(hillock_node_id),
        distal_original_node_id=int(distal_node_id),
        forward_neighbor_original_node_id=int(forward_neighbor),
    )


def prune_graph_from_hillock(
    graph: GraphResult,
    *,
    hillock_node_id: int,
    distal_node_id: int,
) -> HillockPruneResult:
    base_graph = clone_graph(graph)
    node_count = len(base_graph.nodes)
    if not (0 <= hillock_node_id < node_count):
        raise ValueError(f"Hillock node is out of range: {hillock_node_id}")
    if not (0 <= distal_node_id < node_count):
        raise ValueError(f"Distal node is out of range: {distal_node_id}")
    if hillock_node_id == distal_node_id:
        raise ValueError("Hillock node and distal node must be different.")

    path = _path_between(node_count, base_graph.edges, hillock_node_id, distal_node_id)
    if not path or len(path) < 2:
        raise ValueError("Could not find a neurite path from the hillock node to the distal node.")
    forward_neighbor = int(path[1])

    adjacency = _adjacency(node_count, base_graph.edges)
    kept_ids: Set[int] = {int(hillock_node_id)}
    visited: Set[int] = {int(hillock_node_id)}
    stack = [forward_neighbor]
    while stack:
        current = int(stack.pop())
        if current in visited:
            continue
        visited.add(current)
        kept_ids.add(current)
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                stack.append(int(neighbor))

    kept_original_ids = sorted(kept_ids)
    excluded_original_ids = [idx for idx in range(node_count) if idx not in kept_ids]
    original_to_current = {orig_id: new_id for new_id, orig_id in enumerate(kept_original_ids)}
    current_to_original = list(kept_original_ids)

    pruned_nodes: List[Node] = []
    for new_id, original_id in enumerate(kept_original_ids):
        pruned_nodes.append(replace(base_graph.nodes[original_id], id=new_id))
    pruned_edges = [
        (original_to_current[i], original_to_current[j])
        for i, j in base_graph.edges
        if i in original_to_current and j in original_to_current
    ]
    pruned_graph = label_graph(pruned_nodes, pruned_edges)

    return HillockPruneResult(
        graph=pruned_graph,
        kept_original_node_ids=kept_original_ids,
        excluded_original_node_ids=excluded_original_ids,
        original_to_current=original_to_current,
        current_to_original=current_to_original,
        hillock_original_node_id=int(hillock_node_id),
        hillock_current_node_id=int(original_to_current[hillock_node_id]),
        distal_original_node_id=int(distal_node_id),
        distal_current_node_id=int(original_to_current[distal_node_id]),
        forward_neighbor_original_node_id=int(forward_neighbor),
        forward_neighbor_current_node_id=int(original_to_current[forward_neighbor]),
    )
