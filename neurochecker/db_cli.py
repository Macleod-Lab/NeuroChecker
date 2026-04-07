import argparse
import json
from pathlib import Path

from neurochecker.database import SegmentationDatabase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NeuroChecker PostgreSQL database tools."
    )
    parser.add_argument(
        "--dsn",
        required=True,
        help="PostgreSQL DSN, e.g. postgresql://user:pass@host:5432/neurochecker",
    )
    parser.add_argument(
        "--schema",
        default="public",
        help="PostgreSQL schema to use. Defaults to public.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create the NeuroChecker schema and tables.")

    import_neurotracer = subparsers.add_parser(
        "import-neurotracer",
        help="Import a NeuroTracer neuron mask index into PostgreSQL.",
    )
    import_neurotracer.add_argument("--data-root", required=True, help="Root containing neuron_<ID> folders.")
    import_neurotracer.add_argument("--neuron-id", required=True, help="Neuron id or neuron_<ID> name.")
    import_neurotracer.add_argument("--dataset-name", default=None, help="Optional dataset name override.")

    import_export = subparsers.add_parser(
        "import-export",
        help="Import a NeuroChecker export bundle into PostgreSQL.",
    )
    import_export.add_argument("--export-root", required=True, help="Path to a neurochecker_export_* directory.")
    import_export.add_argument("--dataset-name", default=None, help="Optional dataset name override.")
    import_export.add_argument(
        "--skip-reviews",
        action="store_true",
        help="Import metadata and assignments only, without reviewer verdicts.",
    )

    score = subparsers.add_parser(
        "score",
        help="Query a segment and compute consensus scores.",
    )
    score.add_argument("--neuron", required=True, help="Neuron name, e.g. neuron_Bruce.")
    score.add_argument("--segment", required=True, help="Segment path, e.g. root/branch_1.")
    score.add_argument("--frame", type=int, default=None, help="Optional frame filter.")
    score.add_argument("--step", type=int, default=None, help="Optional step filter.")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    with SegmentationDatabase(args.dsn, schema=args.schema) as db:
        if args.command == "init":
            db.initialize()
            print(f"Initialized NeuroChecker tables in schema '{args.schema}'.")
            return 0

        if args.command == "import-neurotracer":
            db.initialize()
            result = db.import_neurotracer_neuron(
                data_root=Path(args.data_root),
                neuron_id=args.neuron_id,
                dataset_name=args.dataset_name,
            )
            print(json.dumps(result, indent=2))
            return 0

        if args.command == "import-export":
            db.initialize()
            result = db.import_export_bundle(
                export_root=Path(args.export_root),
                dataset_name=args.dataset_name,
                include_reviews=not args.skip_reviews,
            )
            print(json.dumps(result, indent=2))
            return 0

        if args.command == "score":
            result = db.compare_segment_scoring(
                neuron_name=args.neuron,
                segment_path=args.segment,
                frame=args.frame,
                step=args.step,
            )
            print(json.dumps(result, indent=2))
            return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
