import colorsys
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from neurochecker.graph import GraphResult, Node, label_graph
from neurochecker.gui.constants import (
    MESH_SKELETON_MAX_VOXELS,
    MESH_SKELETON_MIN_PITCH,
    MESH_SKELETON_TARGET_MAX_DIM,
    _MESH_SKELETON_CACHE,
    logger,
)


def _load_ascii_ply_mesh(path: Path, *, max_faces: int = 200000) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            line = handle.readline().strip()
            if line != "ply":
                logger.debug("PLY header missing in %s", path)
                return None
            fmt = None
            vertex_count = 0
            face_count = 0
            while True:
                line = handle.readline()
                if not line:
                    return None
                line = line.strip()
                if line.startswith("format"):
                    fmt = line.split()[1]
                elif line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("element face"):
                    face_count = int(line.split()[-1])
                elif line.startswith("end_header"):
                    break
            if fmt != "ascii":
                logger.debug("PLY format not ascii in %s (fmt=%s)", path, fmt)
                return None
            verts = np.zeros((vertex_count, 3), dtype=np.float32)
            for i in range(vertex_count):
                parts = handle.readline().strip().split()
                if len(parts) < 3:
                    logger.debug("PLY vertex parse failed in %s at row %d", path, i)
                    return None
                verts[i, 0] = float(parts[0])
                verts[i, 1] = float(parts[1])
                verts[i, 2] = float(parts[2])
            stride = max(1, face_count // max_faces) if max_faces > 0 else 1
            faces = []
            for idx in range(face_count):
                parts = handle.readline().strip().split()
                if not parts:
                    continue
                if stride > 1 and (idx % stride) != 0:
                    continue
                count = int(parts[0])
                if count < 3:
                    continue
                indices = [int(x) for x in parts[1 : count + 1]]
                if count == 3:
                    faces.append(indices)
                else:
                    v0 = indices[0]
                    for k in range(1, count - 1):
                        faces.append([v0, indices[k], indices[k + 1]])
            if not faces:
                logger.debug("PLY faces empty in %s", path)
                return None
            return verts, np.asarray(faces, dtype=np.int32)
    except Exception:
        logger.exception("PLY load failed for %s", path)
        return None


def _mesh_skeleton_cache_path(mesh_path: Path) -> Path:
    return mesh_path.with_suffix(".skeleton.npz")


def _auto_xy_pitch(bounds: np.ndarray, slice_thickness_z: float) -> float:
    """Compute an XY voxel pitch given that Z pitch is fixed to *slice_thickness_z*."""
    bbox = bounds[1] - bounds[0]
    xy_max = float(max(bbox[0], bbox[1])) if bbox.size >= 2 else 0.0
    if xy_max <= 0:
        return MESH_SKELETON_MIN_PITCH
    z_extent = float(bbox[2]) if bbox.size >= 3 else 0.0
    nz = max(1, int(np.ceil(z_extent / slice_thickness_z)) + 1)
    pitch_xy = max(xy_max / float(MESH_SKELETON_TARGET_MAX_DIM), MESH_SKELETON_MIN_PITCH)
    nx = int(np.ceil(float(bbox[0]) / pitch_xy)) + 1
    ny = int(np.ceil(float(bbox[1]) / pitch_xy)) + 1
    total = nx * ny * nz
    if total > MESH_SKELETON_MAX_VOXELS:
        xy_budget = MESH_SKELETON_MAX_VOXELS / float(nz)
        scale = np.sqrt(float(nx * ny) / xy_budget)
        pitch_xy *= scale
    return float(pitch_xy)


def _load_mesh_skeleton_cache(
    cache_path: Path,
    *,
    slice_thickness_z: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path)
    except Exception:
        logger.warning("Mesh skeleton cache load failed: %s", cache_path)
        return None
    if "slice_thickness_z" in data:
        if abs(float(data["slice_thickness_z"]) - slice_thickness_z) > 1e-6:
            return None
    else:
        return None
    points = data.get("points")
    edges = data.get("edges")
    if points is None or edges is None:
        return None
    return points, edges


_MAX_ANISOTROPY_RATIO = 4.0


def _build_mesh_skeleton(
    mesh_path: Path,
    *,
    pitch_xy: Optional[float],
    slice_thickness_z: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Build a skeleton with anisotropic voxelization.

    The mesh is scaled so that isotropic pitch=1 voxelization yields
    *pitch_xy* resolution in XY and *slice_thickness_z* in Z (one voxel
    layer per frame).  If the Z-to-XY ratio exceeds ``_MAX_ANISOTROPY_RATIO``
    the Z pitch is coarsened to keep voxelization feasible; skeleton Z
    coordinates are still returned in real (physical) units.
    """
    try:
        import trimesh
        from skimage import morphology
    except Exception as exc:
        logger.warning("Mesh skeleton dependencies missing: %s", exc)
        return None
    try:
        mesh = trimesh.load(mesh_path, process=False)
    except Exception:
        logger.exception("Mesh load failed for skeleton: %s", mesh_path)
        return None
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            logger.warning("Mesh scene is empty for skeleton: %s", mesh_path)
            return None
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.is_empty:
        logger.warning("Mesh is empty for skeleton: %s", mesh_path)
        return None

    if pitch_xy is None:
        pitch_xy = _auto_xy_pitch(mesh.bounds, slice_thickness_z)

    voxel_z = slice_thickness_z
    ratio = pitch_xy / voxel_z if voxel_z > 0 else 1.0
    if ratio > _MAX_ANISOTROPY_RATIO:
        voxel_z = pitch_xy / _MAX_ANISOTROPY_RATIO
        logger.info(
            "Capping Z voxel pitch: slice_z=%.6f -> voxel_z=%.6f (ratio %.1f -> %.1f)",
            slice_thickness_z, voxel_z, ratio, _MAX_ANISOTROPY_RATIO,
        )

    bbox = mesh.bounds[1] - mesh.bounds[0]
    nz = max(1, int(np.ceil(float(bbox[2]) / voxel_z)) + 1)
    nx = max(1, int(np.ceil(float(bbox[0]) / pitch_xy)) + 1)
    ny = max(1, int(np.ceil(float(bbox[1]) / pitch_xy)) + 1)
    voxels = nx * ny * nz
    logger.info(
        "Mesh skeleton: pitch_xy=%.6f voxel_z=%.6f dims=(%d, %d, %d) voxels=%d",
        pitch_xy, voxel_z, nx, ny, nz, voxels,
    )

    scale = np.array([1.0 / pitch_xy, 1.0 / pitch_xy, 1.0 / voxel_z])
    transform = np.eye(4)
    transform[0, 0] = scale[0]
    transform[1, 1] = scale[1]
    transform[2, 2] = scale[2]
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_transform(transform)

    vg = None
    for method in ("ray", "subdivide", "binvox"):
        try:
            vg = scaled_mesh.voxelized(pitch=1.0, method=method)
            logger.info("Mesh voxelized with method=%s", method)
            break
        except Exception as vox_exc:
            logger.warning("Voxelization method=%s failed: %s", method, vox_exc)
    if vg is None:
        logger.warning(
            "Trimesh voxelization failed; falling back to vertex sampling for %s",
            mesh_path,
        )
        return _vertex_sample_skeleton(
            mesh, mesh_path, pitch_xy=pitch_xy, slice_thickness_z=slice_thickness_z
        )
    vg = vg.fill()
    matrix = vg.matrix.astype(bool)
    if hasattr(morphology, "skeletonize_3d"):
        skel = morphology.skeletonize_3d(matrix)
    else:
        skel = morphology.skeletonize(matrix)
    indices = np.argwhere(skel)
    if indices.size == 0:
        logger.warning("Mesh skeleton empty after thinning: %s", mesh_path)
        return None

    idx_map = {tuple(idx): i for i, idx in enumerate(indices)}
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    edges = set()
    for i, idx in enumerate(indices):
        ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
        for dx, dy, dz in neighbor_offsets:
            nbr = (ix + dx, iy + dy, iz + dz)
            j = idx_map.get(nbr)
            if j is not None and i < j:
                edges.add((i, j))

    points_scaled = trimesh.transform_points(indices.astype(float), vg.transform)
    inv_scale = np.array([pitch_xy, pitch_xy, voxel_z])
    points = (points_scaled * inv_scale).astype(np.float32)

    edges_arr = np.asarray(list(edges), dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)

    if edges_arr.size > 0:
        n_pts = len(points)
        neighbors: List[List[int]] = [[] for _ in range(n_pts)]
        for ei, ej in edges_arr:
            neighbors[ei].append(ej)
            neighbors[ej].append(ei)
        degree = np.array([len(nb) for nb in neighbors], dtype=np.int32)
        is_branch_or_end = (degree != 2)
        lam = 0.4
        for _iteration in range(5):
            new_pts = points.copy()
            for i in range(n_pts):
                if is_branch_or_end[i]:
                    continue
                nb = neighbors[i]
                avg = points[nb].mean(axis=0)
                new_pts[i] = points[i] + lam * (avg - points[i])
            points = new_pts

    return points, edges_arr, float(pitch_xy), float(slice_thickness_z)


def _vertex_sample_skeleton(
    mesh: "trimesh.Trimesh",
    mesh_path: Path,
    *,
    pitch_xy: float,
    slice_thickness_z: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Fallback skeleton from mesh vertices when voxelization fails.

    Groups vertices into a grid, picks one representative per cell, then
    connects neighbors with a KD-tree to form a skeleton graph.
    """
    from scipy.spatial import cKDTree

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return None

    cell = np.array([pitch_xy, pitch_xy, slice_thickness_z])
    grid_idx = np.floor((verts - verts.min(axis=0)) / cell).astype(np.int64)

    cell_map: Dict[tuple, List[int]] = {}
    for i, key in enumerate(map(tuple, grid_idx)):
        cell_map.setdefault(key, []).append(i)

    points_list: List[np.ndarray] = []
    for indices in cell_map.values():
        points_list.append(verts[indices].mean(axis=0))
    points = np.asarray(points_list, dtype=np.float32)
    if len(points) < 2:
        return None

    tree = cKDTree(points / cell)
    pairs = tree.query_pairs(r=1.8)
    edges_arr = (
        np.asarray(list(pairs), dtype=np.int32)
        if pairs
        else np.zeros((0, 2), dtype=np.int32)
    )

    logger.info(
        "Vertex-sample skeleton: %d points, %d edges for %s",
        len(points), len(edges_arr), mesh_path,
    )
    return points, edges_arr, float(pitch_xy), float(slice_thickness_z)


def _mesh_skeleton_graph(
    mesh_path: Path,
    *,
    pixel_size_xy: float,
    slice_thickness_z: float,
) -> Optional[GraphResult]:
    try:
        mesh_mtime = mesh_path.stat().st_mtime
    except OSError:
        mesh_mtime = 0.0
    px = float(pixel_size_xy) if pixel_size_xy > 0 else 1.0
    sz = float(slice_thickness_z) if slice_thickness_z > 0 else 1.0

    cache_key = (
        str(mesh_path),
        float(mesh_mtime),
        px,
        sz,
    )
    cached = _MESH_SKELETON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_path = _mesh_skeleton_cache_path(mesh_path)
    points_edges = None
    if cache_path.exists() and mesh_mtime > 0:
        try:
            if cache_path.stat().st_mtime >= mesh_mtime:
                points_edges = _load_mesh_skeleton_cache(
                    cache_path, slice_thickness_z=sz,
                )
        except OSError:
            pass
    if points_edges is None and cache_path.exists():
        points_edges = _load_mesh_skeleton_cache(
            cache_path, slice_thickness_z=sz,
        )
    if points_edges is None:
        built = _build_mesh_skeleton(
            mesh_path,
            pitch_xy=None,
            slice_thickness_z=sz,
        )
        if built is None:
            return None
        points, edges_arr, stored_pxy, stored_sz = built
        logger.info("Mesh skeleton built: points=%d edges=%d", points.shape[0], edges_arr.shape[0])
        try:
            np.savez_compressed(
                cache_path,
                points=points,
                edges=edges_arr,
                pitch_xy=np.float64(stored_pxy),
                slice_thickness_z=np.float64(stored_sz),
            )
        except Exception:
            logger.warning("Mesh skeleton cache write failed: %s", cache_path)
    else:
        points, edges_arr = points_edges
        logger.info("Mesh skeleton cache loaded: %s", cache_path)

    if points.size == 0:
        return None
    nodes: List[Node] = []
    for i, (x, y, z) in enumerate(points):
        frame = int(round(float(z) / sz))
        nodes.append(
            Node(
                id=i,
                frame=frame,
                run_id="mesh",
                x_px=float(x) / px,
                y_px=float(y) / px,
                z_frame=frame,
                x=float(x),
                y=float(y),
                z=float(z),
                mask_path=str(mesh_path),
                mask_x=0,
                mask_y=0,
                mask_width=0,
                mask_height=0,
                color=None,
            )
        )
    edges = [(int(i), int(j)) for i, j in edges_arr.tolist()] if edges_arr.size else []
    labeled = label_graph(nodes, edges)
    _MESH_SKELETON_CACHE[cache_key] = labeled
    return labeled



def _segment_color_for_plot(segment_id: int) -> str:
    rng = random.Random(segment_id + 1337)
    hue = rng.random()
    sat = 0.65
    val = 0.92
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
