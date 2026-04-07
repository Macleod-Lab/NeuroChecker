import argparse
from pathlib import Path

from neurochecker.pipeline import resolve_data_root, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NeuroChecker: build a skeleton graph from NeuroTracer masks or mesh skeletons."
    )
    parser.add_argument("--neuron-id", required=True, help="Neuron id (e.g. 0 or neuron_0).")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to NeuroTracer output root that contains neuron_<ID> folders.",
    )
    parser.add_argument(
        "--skeleton-path",
        default=None,
        help="Path to mesh skeleton JSON file (alternative to --data-root).",
    )
    parser.add_argument(
        "--mesh-path",
        default=None,
        help="Path to PLY mesh file (for visualization, used with --skeleton-path).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Output directory for reports and skeletons.",
    )
    parser.add_argument(
        "--pixel-size-xy",
        type=float,
        default=0.00495,
        help="Physical pixel size in XY (same units as output).",
    )
    parser.add_argument(
        "--slice-thickness-z",
        type=float,
        default=0.059,
        help="Physical spacing between slices (same units as output).",
    )
    parser.add_argument(
        "--graph-mode",
        choices=("mst", "frame_link"),
        default="mst",
        help="Graph mode: mst (kNN + MST) or frame_link (link to previous frame).",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=6,
        help="Number of neighbors for MST candidate edges.",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=3,
        help="Maximum frame gap for MST candidate edges.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Maximum physical edge length (omit for no limit).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate input arguments
    if not args.data_root and not args.skeleton_path:
        parser.error("Must specify either --data-root (for NeuroTracer data) or --skeleton-path (for mesh skeleton)")

    if args.data_root and args.skeleton_path:
        parser.error("Cannot specify both --data-root and --skeleton-path")

    data_root = None
    if args.data_root:
        data_root = resolve_data_root(args.data_root, args.neuron_id)

    skeleton_path = Path(args.skeleton_path) if args.skeleton_path else None
    mesh_path = Path(args.mesh_path) if args.mesh_path else None
    out_dir = Path(args.out_dir)

    output_root, _ = run_pipeline(
        neuron_id=args.neuron_id,
        data_root=data_root,
        skeleton_path=skeleton_path,
        mesh_path=mesh_path,
        out_dir=out_dir,
        pixel_size_xy=args.pixel_size_xy,
        slice_thickness_z=args.slice_thickness_z,
        graph_mode=args.graph_mode,
        k_neighbors=args.k_neighbors,
        max_frame_gap=args.max_frame_gap,
        max_distance=args.max_distance,
    )

    print(f"NeuroChecker outputs written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
