import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from neurochecker.graph import build_graph, build_nodes, GraphResult, Node
from neurochecker.mask_io import collect_run_stats, load_mask_entries


def normalize_neuron_id(neuron_id: str) -> str:
    neuron_id = str(neuron_id).strip()
    if neuron_id.lower().startswith("neuron_"):
        neuron_id = neuron_id[7:]
    return neuron_id


def resolve_data_root(data_root: Optional[str], neuron_id: str) -> Path:
    neuron_id = normalize_neuron_id(neuron_id)
    if data_root:
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        return root

    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent / "neurotracer-standalone",
    ]
    for candidate in candidates:
        if (candidate / f"neuron_{neuron_id}").exists():
            return candidate
    raise FileNotFoundError(
        "Could not auto-detect data root. Pass --data-root pointing to NeuroTracer output."
    )


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_mesh_skeleton(skeleton_path: Path) -> Tuple[List[Node], Dict[str, object]]:
    """Load skeleton data from mesh skeleton JSON file."""
    with skeleton_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes_data = data["nodes"]
    node_pixel_coords = data.get("node_pixel_coords", [])
    node_frames = data.get("node_frames", [])
    pixel_size_xy = data.get("pixel_size_xy", 1.0)
    slice_thickness_z = data.get("slice_thickness_z", 1.0)
    
    nodes = []
    for i, (coords, pixel_coord, frame) in enumerate(zip(nodes_data, node_pixel_coords, node_frames)):
        x, y, z = coords
        x_px, y_px = pixel_coord
        
        node = Node(
            id=i,
            frame=frame,
            run_id="mesh",
            x_px=x_px,
            y_px=y_px,
            z_frame=frame,  # Use frame as z_frame for now
            x=x,
            y=y,
            z=z,
            degree=0,  # Will be calculated later
            label=f"node_{i}",
            mask_path="",  # No original mask
            mask_x=0,
            mask_y=0,
            mask_width=0,
            mask_height=0,
            color=None,
        )
        nodes.append(node)

    # Calculate degrees
    edges = data.get("edges", [])
    degree_count = {}
    for edge in edges:
        for node_id in edge:
            degree_count[node_id] = degree_count.get(node_id, 0) + 1
    
    for node in nodes:
        node.degree = degree_count.get(node.id, 0)

    run_stats = {
        "mesh": SimpleNamespace(
            max_width=max((n.x_px for n in nodes), default=0),
            max_height=max((n.y_px for n in nodes), default=0),
            color=None,
        )
    }

    return nodes, run_stats


def run_pipeline(
    *,
    neuron_id: str,
    data_root: Optional[Path] = None,
    skeleton_path: Optional[Path] = None,
    mesh_path: Optional[Path] = None,
    out_dir: Path,
    pixel_size_xy: float = 0.00495,
    slice_thickness_z: float = 0.059,
    graph_mode: str = "mst",
    k_neighbors: int = 20,
    max_frame_gap: int = 8,
    max_distance: Optional[float] = None,
) -> Tuple[Path, Dict[str, object]]:
    neuron_id = normalize_neuron_id(neuron_id)

    # Load data based on input type
    if skeleton_path and skeleton_path.exists():
        # Load from mesh skeleton JSON
        nodes, run_stats = load_mesh_skeleton(skeleton_path)
        # For mesh skeletons, we already have the graph structure
        # We need to rebuild it or use the existing one
        if "edges" in json.load(skeleton_path.open()):
            # If skeleton has edges, create a GraphResult from it
            skeleton_data = json.load(skeleton_path.open())
            edges = [(i, j) for i, j in skeleton_data["edges"]]
            graph = GraphResult(
                nodes=nodes,
                edges=edges,
                order=list(range(len(nodes))),
                paths=[],  # Could be reconstructed if needed
                counts=skeleton_data.get("counts", {})
            )
        else:
            # Rebuild graph from nodes
            graph = build_graph(
                nodes,
                mode=graph_mode,
                k_neighbors=k_neighbors,
                max_frame_gap=max_frame_gap,
                max_distance=max_distance,
            )
    elif data_root:
        # Traditional NeuroTracer pipeline
        entries = load_mask_entries(data_root, neuron_id)
        nodes = build_nodes(
            entries,
            pixel_size_xy=pixel_size_xy,
            slice_thickness_z=slice_thickness_z,
        )
        run_stats = collect_run_stats(entries)
        graph = build_graph(
            nodes,
            mode=graph_mode,
            k_neighbors=k_neighbors,
            max_frame_gap=max_frame_gap,
            max_distance=max_distance,
        )
    else:
        raise ValueError("Must provide either data_root (for NeuroTracer) or skeleton_path (for mesh skeleton)")

    branch_points = [node.id for node in graph.nodes if node.label == "branch"]
    end_points = [node.id for node in graph.nodes if node.label == "endpoint"]

    output_root = out_dir / f"neuron_{neuron_id}"
    output_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_root / "skeleton.json",
        {
            "neuron_id": neuron_id,
            "spacing": {
                "pixel_size_xy": float(pixel_size_xy),
                "slice_thickness_z": float(slice_thickness_z),
            },
            "graph_mode": graph_mode,
            "nodes": [
                {
                    "id": node.id,
                    "frame": node.frame,
                    "run_id": node.run_id,
                    "x_px": node.x_px,
                    "y_px": node.y_px,
                    "z_frame": node.z_frame,
                    "x": node.x,
                    "y": node.y,
                    "z": node.z,
                    "degree": node.degree,
                    "label": node.label,
                }
                for node in graph.nodes
            ],
            "edges": [[int(i), int(j)] for i, j in graph.edges],
            "paths": graph.paths,
            "branch_points": branch_points,
            "end_points": end_points,
            "skeleton_points": [node.id for node in graph.nodes],
            "counts": graph.counts,
        },
    )

    _write_json(
        output_root / "sample_points.json",
        {
            "neuron_id": neuron_id,
            "spacing": {
                "pixel_size_xy": float(pixel_size_xy),
                "slice_thickness_z": float(slice_thickness_z),
            },
            "points": [
                {
                    "id": node.id,
                    "frame": node.frame,
                    "run_id": node.run_id,
                    "x_px": node.x_px,
                    "y_px": node.y_px,
                    "z_frame": node.z_frame,
                    "x": node.x,
                    "y": node.y,
                    "z": node.z,
                    "label": node.label,
                }
                for node in graph.nodes
            ],
            "branch_points": branch_points,
            "end_points": end_points,
            "skeleton_points": [node.id for node in graph.nodes],
            "counts": graph.counts,
            "runs": {
                run_id: {"max_width": stats.max_width, "max_height": stats.max_height}
                for run_id, stats in run_stats.items()
            },
        },
    )

    return output_root, {"graph": graph}
