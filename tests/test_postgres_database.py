import csv
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from neurochecker import database as database_module
from neurochecker.database import SegmentationDatabase


TEST_DSN = os.getenv("NEUROCHECKER_TEST_DSN")
CAN_RUN_POSTGRES = database_module.psycopg is not None and bool(TEST_DSN)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array > 0).astype(np.uint8) * 255).save(path)


def _write_npz(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mask=array.astype(np.uint8))


def _build_export_fixture(root: Path, *, neuron_name: str, segment_path: str) -> Path:
    export_root = root / f"export_{uuid.uuid4().hex[:8]}"
    export_root.mkdir(parents=True, exist_ok=True)

    mask_a = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    mask_b = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    raw_mask_0 = root / "raw" / "run_demo_00010_a.npz"
    raw_mask_1 = root / "raw" / "run_demo_00011_b.npz"
    _write_npz(raw_mask_0, mask_a)
    _write_npz(raw_mask_1, mask_b)

    image_0 = export_root / "segments" / "root" / "branch_1" / "images" / "frame_00010_step_00000.png"
    image_1 = export_root / "segments" / "root" / "branch_1" / "images" / "frame_00011_step_00001.png"
    mask_0 = export_root / "segments" / "root" / "branch_1" / "masks" / "frame_00010_step_00000_mask.png"
    mask_1 = export_root / "segments" / "root" / "branch_1" / "masks" / "frame_00011_step_00001_mask.png"
    _write_mask(image_0, np.ones((4, 4), dtype=np.uint8))
    _write_mask(image_1, np.ones((4, 4), dtype=np.uint8))
    _write_mask(mask_0, mask_a)
    _write_mask(mask_1, mask_b)

    sample_rows = [
        {
            "neuron": neuron_name,
            "segment": segment_path,
            "segment_index": 1,
            "node_id": 0,
            "frame": 10,
            "step": 0,
            "image_path": str(image_0),
            "mask_path": str(mask_0),
            "run_id": "run_demo",
            "mask_id": "run_demo_00010_a",
            "centerline_x_px": 100.0,
            "centerline_y_px": 200.0,
            "centerline_x": 1.0,
            "centerline_y": 2.0,
            "centerline_z": 3.0,
            "centroid_x_px": 101.0,
            "centroid_y_px": 201.0,
            "sample_center_x_px": 100.5,
            "sample_center_y_px": 200.5,
            "crop_x": 90,
            "crop_y": 190,
            "crop_w": 32,
            "crop_h": 32,
            "scale_factor": 1.0,
            "has_mask": 1,
            "flagged": 0,
        },
        {
            "neuron": neuron_name,
            "segment": segment_path,
            "segment_index": 1,
            "node_id": 1,
            "frame": 11,
            "step": 1,
            "image_path": str(image_1),
            "mask_path": str(mask_1),
            "run_id": "run_demo",
            "mask_id": "run_demo_00011_b",
            "centerline_x_px": 102.0,
            "centerline_y_px": 202.0,
            "centerline_x": 1.2,
            "centerline_y": 2.2,
            "centerline_z": 3.2,
            "centroid_x_px": 103.0,
            "centroid_y_px": 203.0,
            "sample_center_x_px": 102.5,
            "sample_center_y_px": 202.5,
            "crop_x": 91,
            "crop_y": 191,
            "crop_w": 32,
            "crop_h": 32,
            "scale_factor": 1.0,
            "has_mask": 1,
            "flagged": 0,
        },
    ]
    with (export_root / "sample_points.json").open("w", encoding="utf-8") as handle:
        json.dump(sample_rows, handle, indent=2)

    metadata_rows = [
        {
            "neuron": neuron_name,
            "segment": segment_path,
            "segment_index": 1,
            "node_id": 0,
            "frame": 10,
            "step": 0,
            "run_id": "run_demo",
            "mask_id": "run_demo_00010_a",
            "mask_path": str(mask_0),
            "image_path": str(image_0),
            "crop_x": 90,
            "crop_y": 190,
            "crop_w": 32,
            "crop_h": 32,
            "scale_factor": 1.0,
            "centerline_x_px": 100.0,
            "centerline_y_px": 200.0,
            "centroid_x_px": 101.0,
            "centroid_y_px": 201.0,
            "sample_center_x_px": 100.5,
            "sample_center_y_px": 200.5,
            "mask_x": 99,
            "mask_y": 199,
            "mask_w": 4,
            "mask_h": 4,
            "full_width": 512,
            "full_height": 512,
            "flagged": 0,
        },
        {
            "neuron": neuron_name,
            "segment": segment_path,
            "segment_index": 1,
            "node_id": 1,
            "frame": 11,
            "step": 1,
            "run_id": "run_demo",
            "mask_id": "run_demo_00011_b",
            "mask_path": str(mask_1),
            "image_path": str(image_1),
            "crop_x": 91,
            "crop_y": 191,
            "crop_w": 32,
            "crop_h": 32,
            "scale_factor": 1.0,
            "centerline_x_px": 102.0,
            "centerline_y_px": 202.0,
            "centroid_x_px": 103.0,
            "centroid_y_px": 203.0,
            "sample_center_x_px": 102.5,
            "sample_center_y_px": 202.5,
            "mask_x": 100,
            "mask_y": 200,
            "mask_w": 4,
            "mask_h": 4,
            "full_width": 512,
            "full_height": 512,
            "flagged": 0,
        },
    ]
    _write_csv(export_root / "metadata.csv", metadata_rows)

    flags_rows = [
        {
            "mask_id": "run_demo_00010_a",
            "run_id": "run_demo",
            "frame": 10,
            "x": 99,
            "y": 199,
            "width": 0,
            "height": 0,
            "full_width": 512,
            "full_height": 512,
            "path": str(raw_mask_0),
            "flagged": 0,
        },
        {
            "mask_id": "run_demo_00011_b",
            "run_id": "run_demo",
            "frame": 11,
            "x": 100,
            "y": 200,
            "width": 0,
            "height": 0,
            "full_width": 512,
            "full_height": 512,
            "path": str(raw_mask_1),
            "flagged": 0,
        },
    ]
    _write_csv(export_root / "flags.csv", flags_rows)

    skeleton = {
        "nodes": [[1.0, 2.0, 3.0], [1.2, 2.2, 3.2]],
        "node_pixel_coords": [[100.0, 200.0], [102.0, 202.0]],
        "node_frames": [10, 11],
        "edges": [[0, 1]],
        "segments": [[0, 1]],
        "node_to_segments": {"0": [1], "1": [1]},
        "pixel_size_xy": 0.00495,
        "slice_thickness_z": 0.059,
    }
    with (export_root / "skeleton.json").open("w", encoding="utf-8") as handle:
        json.dump(skeleton, handle, indent=2)

    segment_tree = {
        "root_segment_index": 1,
        "total_segments": 1,
        "segments": {
            "1": {
                "path": segment_path,
                "parent": None,
                "children": [],
                "node_count": 2,
                "frame_count": 2,
            }
        },
    }
    with (export_root / "segment_tree.json").open("w", encoding="utf-8") as handle:
        json.dump(segment_tree, handle, indent=2)

    assignments = [
        {"segment": segment_path, "frame": 10, "step": 0, "reviewer": "alice"},
        {"segment": segment_path, "frame": 10, "step": 0, "reviewer": "bob"},
        {"segment": segment_path, "frame": 10, "step": 0, "reviewer": "carol"},
        {"segment": segment_path, "frame": 11, "step": 1, "reviewer": "alice"},
        {"segment": segment_path, "frame": 11, "step": 1, "reviewer": "bob"},
        {"segment": segment_path, "frame": 11, "step": 1, "reviewer": "carol"},
    ]
    _write_csv(export_root / "assignments.csv", assignments)

    alice_verdicts = {f"{segment_path}/00010/00000": "good", f"{segment_path}/00011/00001": "bad"}
    bob_verdicts = {f"{segment_path}/00010/00000": "good", f"{segment_path}/00011/00001": "bad"}
    carol_verdicts = {f"{segment_path}/00010/00000": "bad", f"{segment_path}/00011/00001": "good"}
    for user, verdicts in {
        "alice": alice_verdicts,
        "bob": bob_verdicts,
        "carol": carol_verdicts,
    }.items():
        user_dir = export_root / "reviews" / user
        user_dir.mkdir(parents=True, exist_ok=True)
        with (user_dir / "verdicts.json").open("w", encoding="utf-8") as handle:
            json.dump(verdicts, handle, indent=2)

    _write_mask(
        export_root / "reviews" / "alice" / "deltas" / "root" / "branch_1" / "frame_00011_mask.png",
        np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
    )
    _write_mask(
        export_root / "reviews" / "bob" / "deltas" / "root" / "branch_1" / "frame_00011_mask.png",
        np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
    )
    _write_mask(
        export_root / "reviews" / "carol" / "deltas" / "root" / "branch_1" / "frame_00010_mask.png",
        np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
    )
    return export_root


@unittest.skipUnless(CAN_RUN_POSTGRES, "Requires psycopg and NEUROCHECKER_TEST_DSN")
class PostgresDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = f"test_neurochecker_{uuid.uuid4().hex[:8]}"
        cls.db = SegmentationDatabase(TEST_DSN, schema=cls.schema)
        cls.db.initialize()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        conn = database_module.psycopg.connect(TEST_DSN)
        try:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{cls.schema}" CASCADE')
            conn.commit()
        finally:
            conn.close()

    def test_import_export_bundle_populates_relational_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            neuron_name = f"neuron_demo_{uuid.uuid4().hex[:6]}"
            export_root = _build_export_fixture(Path(tmpdir), neuron_name=neuron_name, segment_path="root/branch_1")
            result = self.db.import_export_bundle(export_root=export_root, dataset_name=f"dataset_{uuid.uuid4().hex[:6]}")

        self.assertEqual(result["segments"], 1)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["assignments"], 6)
        self.assertEqual(result["reviews"], 6)

        rows = self.db._fetchall("SELECT COUNT(*) AS count FROM source_masks")
        self.assertGreaterEqual(int(rows[0]["count"]), 2)

        segment_samples = self.db.get_segment_samples(neuron_name=neuron_name, segment_path="root/branch_1")
        self.assertEqual(len(segment_samples), 2)
        self.assertEqual(segment_samples[0]["mask_w"], 4)
        self.assertEqual(segment_samples[1]["mask_h"], 4)

    def test_compare_segment_scoring_returns_distinct_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            neuron_name = f"neuron_scores_{uuid.uuid4().hex[:6]}"
            export_root = _build_export_fixture(Path(tmpdir), neuron_name=neuron_name, segment_path="root/branch_1")
            self.db.import_export_bundle(export_root=export_root, dataset_name=f"dataset_{uuid.uuid4().hex[:6]}")

        scores = self.db.compare_segment_scoring(
            neuron_name=neuron_name,
            segment_path="root/branch_1",
        )

        self.assertEqual(len(scores["samples"]), 2)
        frame10 = next(row for row in scores["samples"] if row["frame"] == 10)
        frame11 = next(row for row in scores["samples"] if row["frame"] == 11)

        self.assertAlmostEqual(frame10["acceptance_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(frame10["agreement_ratio"], 2.0 / 3.0)
        self.assertIsNone(frame10["mask_agreement"])

        self.assertAlmostEqual(frame11["acceptance_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(frame11["agreement_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(frame11["mask_agreement"], 1.0 / 3.0)
        self.assertAlmostEqual(frame11["hybrid_consensus"], 0.5)

        summary = scores["summary"]
        self.assertAlmostEqual(summary["avg_acceptance_ratio"], 0.5)
        self.assertAlmostEqual(summary["avg_agreement_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["avg_hybrid_consensus"], (2.0 / 3.0 + 0.5) / 2.0)

    def test_record_review_updates_segment_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            neuron_name = f"neuron_record_{uuid.uuid4().hex[:6]}"
            export_root = _build_export_fixture(Path(tmpdir), neuron_name=neuron_name, segment_path="root/branch_1")
            self.db.import_export_bundle(
                export_root=export_root,
                dataset_name=f"dataset_{uuid.uuid4().hex[:6]}",
                include_reviews=False,
            )

        self.db.record_review(
            neuron_name=neuron_name,
            segment_path="root/branch_1",
            frame=10,
            step=0,
            reviewer="web_user",
            verdict="bad",
            delta_mask=np.array([[1, 1], [0, 0]], dtype=np.uint8),
            metadata={"source": "web"},
        )

        frame10 = self.db.segment_consensus_rows(
            neuron_name=neuron_name,
            segment_path="root/branch_1",
            frame=10,
            step=0,
        )[0]
        self.assertEqual(frame10["num_reviewers"], 4)
        self.assertEqual(frame10["num_bad"], 1)
        self.assertEqual(frame10["num_pending"], 3)
