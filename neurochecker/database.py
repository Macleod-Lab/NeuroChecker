"""PostgreSQL-backed storage for NeuroTracer and NeuroChecker review data."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from importlib.resources import files
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from neurochecker.consensus_metrics import mask_iou, score_bundle

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except Exception:  # pragma: no cover - import is environment-specific
    psycopg = None
    dict_row = None
    Jsonb = None


SCHEMA_RESOURCE = files("neurochecker.sql").joinpath("postgres_schema.sql")
VALID_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonb(payload: Optional[Dict[str, Any]]) -> Any:
    if payload is None:
        return None
    if Jsonb is None:
        return json.dumps(payload, sort_keys=True)
    return Jsonb(payload)


def _normalize_neuron_id(neuron_id: str) -> str:
    value = str(neuron_id).strip()
    if value.lower().startswith("neuron_"):
        value = value[7:]
    return value


def _coerce_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _parse_color(value: Any) -> Optional[Tuple[int, int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return (int(value[0]), int(value[1]), int(value[2]))
    except Exception:
        return None


def _mask_png_bytes(mask: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(buffer, format="PNG")
    return buffer.getvalue()


def _load_mask_from_png_bytes(payload: bytes) -> Optional[np.ndarray]:
    if not payload:
        return None
    try:
        image = Image.open(BytesIO(bytes(payload))).convert("L")
    except Exception:
        return None
    return (np.asarray(image) > 127).astype(np.uint8)


def _load_mask_from_png_path(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        image = Image.open(candidate).convert("L")
    except Exception:
        return None
    return (np.asarray(image) > 127).astype(np.uint8)


def _mask_dimensions(npz_path: Optional[Path]) -> Tuple[int, int]:
    if npz_path is None or not npz_path.exists():
        return (0, 0)
    try:
        with np.load(npz_path) as payload:
            mask = np.asarray(payload["mask"])
    except Exception:
        return (0, 0)
    if mask.ndim < 2:
        return (0, 0)
    return (int(mask.shape[1]), int(mask.shape[0]))


def _parse_frame_key(value: str) -> Tuple[str, int, int]:
    parts = str(value).split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid frame key: {value}")
    segment = "/".join(parts[:-2])
    frame = int(parts[-2])
    step = int(parts[-1])
    return segment, frame, step


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered)) / float(len(filtered))


def _branch_label(segment_path: str) -> str:
    path = PurePosixPath(str(segment_path))
    return path.name or str(segment_path)


class SegmentationDatabase:
    """High-concurrency PostgreSQL store for segmentation review workflows."""

    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        if psycopg is None:
            raise RuntimeError(
                "psycopg is required for PostgreSQL support. Install psycopg[binary] first."
            )
        if not dsn or not str(dsn).startswith(("postgres://", "postgresql://")):
            raise ValueError("Pass a PostgreSQL DSN starting with postgres:// or postgresql://")
        if not VALID_SCHEMA_RE.match(schema):
            raise ValueError(f"Invalid PostgreSQL schema name: {schema}")
        self.dsn = str(dsn)
        self.schema = schema
        self.conn = psycopg.connect(self.dsn, row_factory=dict_row)
        self._prepare_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SegmentationDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.conn.rollback()
        self.close()

    def _prepare_schema(self) -> None:
        ident = f'"{self.schema}"'
        with self.conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {ident}")
            cur.execute(f"SET search_path TO {ident}, public")
        self.conn.commit()

    def initialize(self) -> None:
        schema_sql = SCHEMA_RESOURCE.read_text(encoding="utf-8")
        statements = [stmt.strip() for stmt in schema_sql.split(";") if stmt.strip()]
        with self.conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
        self.conn.commit()

    def _fetchone(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(query, params or {})
            return cur.fetchone()

    def _fetchall(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(query, params or {})
            return list(cur.fetchall())

    def _execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        with self.conn.cursor() as cur:
            cur.execute(query, params or {})

    def _upsert_dataset(self, *, name: str, source_root: Optional[str]) -> int:
        row = self._fetchone(
            """
            INSERT INTO datasets (name, source_root)
            VALUES (%(name)s, %(source_root)s)
            ON CONFLICT (name) DO UPDATE
            SET source_root = COALESCE(EXCLUDED.source_root, datasets.source_root)
            RETURNING id
            """,
            {"name": name, "source_root": source_root},
        )
        assert row is not None
        return int(row["id"])

    def _upsert_neuron(
        self,
        *,
        dataset_id: int,
        name: str,
        external_id: Optional[str],
        source_type: str,
        export_root: Optional[str],
        pixel_size_xy: Optional[float],
        slice_thickness_z: Optional[float],
        raw_json: Optional[Dict[str, Any]],
    ) -> int:
        row = self._fetchone(
            """
            INSERT INTO neurons (
                dataset_id,
                name,
                external_id,
                source_type,
                export_root,
                pixel_size_xy,
                slice_thickness_z,
                raw_json
            )
            VALUES (
                %(dataset_id)s,
                %(name)s,
                %(external_id)s,
                %(source_type)s,
                %(export_root)s,
                %(pixel_size_xy)s,
                %(slice_thickness_z)s,
                %(raw_json)s
            )
            ON CONFLICT (dataset_id, name) DO UPDATE
            SET external_id = COALESCE(EXCLUDED.external_id, neurons.external_id),
                source_type = EXCLUDED.source_type,
                export_root = COALESCE(EXCLUDED.export_root, neurons.export_root),
                pixel_size_xy = COALESCE(EXCLUDED.pixel_size_xy, neurons.pixel_size_xy),
                slice_thickness_z = COALESCE(EXCLUDED.slice_thickness_z, neurons.slice_thickness_z),
                raw_json = COALESCE(EXCLUDED.raw_json, neurons.raw_json)
            RETURNING id
            """,
            {
                "dataset_id": dataset_id,
                "name": name,
                "external_id": external_id,
                "source_type": source_type,
                "export_root": export_root,
                "pixel_size_xy": pixel_size_xy,
                "slice_thickness_z": slice_thickness_z,
                "raw_json": _jsonb(raw_json),
            },
        )
        assert row is not None
        return int(row["id"])

    def _upsert_run(
        self,
        *,
        neuron_id: int,
        run_id: str,
        direction: Optional[str],
        run_started: Optional[str],
        color: Optional[Tuple[int, int, int]],
        max_width: int,
        max_height: int,
        raw_json: Optional[Dict[str, Any]],
    ) -> int:
        row = self._fetchone(
            """
            INSERT INTO segmentation_runs (
                neuron_id,
                run_id,
                direction,
                run_started,
                color_r,
                color_g,
                color_b,
                max_width,
                max_height,
                raw_json
            )
            VALUES (
                %(neuron_id)s,
                %(run_id)s,
                %(direction)s,
                %(run_started)s,
                %(color_r)s,
                %(color_g)s,
                %(color_b)s,
                %(max_width)s,
                %(max_height)s,
                %(raw_json)s
            )
            ON CONFLICT (neuron_id, run_id) DO UPDATE
            SET direction = COALESCE(EXCLUDED.direction, segmentation_runs.direction),
                run_started = COALESCE(EXCLUDED.run_started, segmentation_runs.run_started),
                color_r = COALESCE(EXCLUDED.color_r, segmentation_runs.color_r),
                color_g = COALESCE(EXCLUDED.color_g, segmentation_runs.color_g),
                color_b = COALESCE(EXCLUDED.color_b, segmentation_runs.color_b),
                max_width = GREATEST(segmentation_runs.max_width, EXCLUDED.max_width),
                max_height = GREATEST(segmentation_runs.max_height, EXCLUDED.max_height),
                raw_json = COALESCE(EXCLUDED.raw_json, segmentation_runs.raw_json)
            RETURNING id
            """,
            {
                "neuron_id": neuron_id,
                "run_id": run_id,
                "direction": direction,
                "run_started": run_started,
                "color_r": None if color is None else color[0],
                "color_g": None if color is None else color[1],
                "color_b": None if color is None else color[2],
                "max_width": int(max_width),
                "max_height": int(max_height),
                "raw_json": _jsonb(raw_json),
            },
        )
        assert row is not None
        return int(row["id"])

    def _upsert_source_mask(
        self,
        *,
        neuron_id: int,
        run_db_id: Optional[int],
        mask_id: str,
        frame: int,
        x: int,
        y: int,
        width: int,
        height: int,
        full_width: int,
        full_height: int,
        path: Optional[str],
        color: Optional[Tuple[int, int, int]],
        flagged: bool,
        raw_json: Optional[Dict[str, Any]],
    ) -> int:
        row = self._fetchone(
            """
            INSERT INTO source_masks (
                neuron_id,
                run_db_id,
                mask_id,
                frame,
                x,
                y,
                width,
                height,
                full_width,
                full_height,
                path,
                color_r,
                color_g,
                color_b,
                flagged,
                raw_json
            )
            VALUES (
                %(neuron_id)s,
                %(run_db_id)s,
                %(mask_id)s,
                %(frame)s,
                %(x)s,
                %(y)s,
                %(width)s,
                %(height)s,
                %(full_width)s,
                %(full_height)s,
                %(path)s,
                %(color_r)s,
                %(color_g)s,
                %(color_b)s,
                %(flagged)s,
                %(raw_json)s
            )
            ON CONFLICT (neuron_id, mask_id) DO UPDATE
            SET run_db_id = COALESCE(EXCLUDED.run_db_id, source_masks.run_db_id),
                frame = EXCLUDED.frame,
                x = EXCLUDED.x,
                y = EXCLUDED.y,
                width = GREATEST(source_masks.width, EXCLUDED.width),
                height = GREATEST(source_masks.height, EXCLUDED.height),
                full_width = GREATEST(source_masks.full_width, EXCLUDED.full_width),
                full_height = GREATEST(source_masks.full_height, EXCLUDED.full_height),
                path = COALESCE(EXCLUDED.path, source_masks.path),
                color_r = COALESCE(EXCLUDED.color_r, source_masks.color_r),
                color_g = COALESCE(EXCLUDED.color_g, source_masks.color_g),
                color_b = COALESCE(EXCLUDED.color_b, source_masks.color_b),
                flagged = source_masks.flagged OR EXCLUDED.flagged,
                raw_json = COALESCE(EXCLUDED.raw_json, source_masks.raw_json)
            RETURNING id
            """,
            {
                "neuron_id": neuron_id,
                "run_db_id": run_db_id,
                "mask_id": mask_id,
                "frame": int(frame),
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "full_width": int(full_width),
                "full_height": int(full_height),
                "path": path,
                "color_r": None if color is None else color[0],
                "color_g": None if color is None else color[1],
                "color_b": None if color is None else color[2],
                "flagged": bool(flagged),
                "raw_json": _jsonb(raw_json),
            },
        )
        assert row is not None
        return int(row["id"])

    def _upsert_segment(
        self,
        *,
        neuron_id: int,
        segment_index: int,
        segment_path: str,
        parent_segment_index: Optional[int],
        node_count: int,
        frame_count: int,
        raw_json: Optional[Dict[str, Any]],
    ) -> int:
        row = self._fetchone(
            """
            INSERT INTO segments (
                neuron_id,
                segment_index,
                segment_path,
                branch_label,
                parent_segment_index,
                node_count,
                frame_count,
                raw_json
            )
            VALUES (
                %(neuron_id)s,
                %(segment_index)s,
                %(segment_path)s,
                %(branch_label)s,
                %(parent_segment_index)s,
                %(node_count)s,
                %(frame_count)s,
                %(raw_json)s
            )
            ON CONFLICT (neuron_id, segment_index) DO UPDATE
            SET segment_path = EXCLUDED.segment_path,
                branch_label = EXCLUDED.branch_label,
                parent_segment_index = COALESCE(EXCLUDED.parent_segment_index, segments.parent_segment_index),
                node_count = GREATEST(segments.node_count, EXCLUDED.node_count),
                frame_count = GREATEST(segments.frame_count, EXCLUDED.frame_count),
                raw_json = COALESCE(EXCLUDED.raw_json, segments.raw_json)
            RETURNING id
            """,
            {
                "neuron_id": neuron_id,
                "segment_index": int(segment_index),
                "segment_path": segment_path,
                "branch_label": _branch_label(segment_path),
                "parent_segment_index": parent_segment_index,
                "node_count": int(node_count),
                "frame_count": int(frame_count),
                "raw_json": _jsonb(raw_json),
            },
        )
        assert row is not None
        return int(row["id"])

    def _upsert_skeleton_node(
        self,
        *,
        neuron_id: int,
        node_index: int,
        frame: int,
        x: Optional[float],
        y: Optional[float],
        z: Optional[float],
        x_px: Optional[float],
        y_px: Optional[float],
        label: Optional[str],
        degree: Optional[int],
        run_id: Optional[str],
        raw_json: Optional[Dict[str, Any]],
    ) -> int:
        row = self._fetchone(
            """
            INSERT INTO skeleton_nodes (
                neuron_id,
                node_index,
                frame,
                x,
                y,
                z,
                x_px,
                y_px,
                label,
                degree,
                run_id,
                raw_json
            )
            VALUES (
                %(neuron_id)s,
                %(node_index)s,
                %(frame)s,
                %(x)s,
                %(y)s,
                %(z)s,
                %(x_px)s,
                %(y_px)s,
                %(label)s,
                %(degree)s,
                %(run_id)s,
                %(raw_json)s
            )
            ON CONFLICT (neuron_id, node_index) DO UPDATE
            SET frame = EXCLUDED.frame,
                x = COALESCE(EXCLUDED.x, skeleton_nodes.x),
                y = COALESCE(EXCLUDED.y, skeleton_nodes.y),
                z = COALESCE(EXCLUDED.z, skeleton_nodes.z),
                x_px = COALESCE(EXCLUDED.x_px, skeleton_nodes.x_px),
                y_px = COALESCE(EXCLUDED.y_px, skeleton_nodes.y_px),
                label = COALESCE(EXCLUDED.label, skeleton_nodes.label),
                degree = COALESCE(EXCLUDED.degree, skeleton_nodes.degree),
                run_id = COALESCE(EXCLUDED.run_id, skeleton_nodes.run_id),
                raw_json = COALESCE(EXCLUDED.raw_json, skeleton_nodes.raw_json)
            RETURNING id
            """,
            {
                "neuron_id": neuron_id,
                "node_index": int(node_index),
                "frame": int(frame),
                "x": x,
                "y": y,
                "z": z,
                "x_px": x_px,
                "y_px": y_px,
                "label": label,
                "degree": degree,
                "run_id": run_id,
                "raw_json": _jsonb(raw_json),
            },
        )
        assert row is not None
        return int(row["id"])

    def _insert_skeleton_edge(self, *, neuron_id: int, source_node_db_id: int, target_node_db_id: int) -> None:
        left = min(int(source_node_db_id), int(target_node_db_id))
        right = max(int(source_node_db_id), int(target_node_db_id))
        self._execute(
            """
            INSERT INTO skeleton_edges (neuron_id, source_node_db_id, target_node_db_id)
            VALUES (%(neuron_id)s, %(source_node_db_id)s, %(target_node_db_id)s)
            ON CONFLICT (neuron_id, source_node_db_id, target_node_db_id) DO NOTHING
            """,
            {
                "neuron_id": neuron_id,
                "source_node_db_id": left,
                "target_node_db_id": right,
            },
        )

    def _insert_segment_node(self, *, segment_id: int, node_db_id: int, node_order: int) -> None:
        self._execute(
            """
            INSERT INTO segment_nodes (segment_id, node_db_id, node_order)
            VALUES (%(segment_id)s, %(node_db_id)s, %(node_order)s)
            ON CONFLICT (segment_id, node_db_id) DO UPDATE
            SET node_order = EXCLUDED.node_order
            """,
            {
                "segment_id": segment_id,
                "node_db_id": node_db_id,
                "node_order": int(node_order),
            },
        )

    def _upsert_segment_sample(
        self,
        *,
        neuron_id: int,
        segment_id: int,
        node_db_id: Optional[int],
        run_db_id: Optional[int],
        source_mask_id: Optional[int],
        run_id: Optional[str],
        mask_id: Optional[str],
        frame: int,
        step: int,
        image_path: str,
        mask_path: Optional[str],
        has_mask: bool,
        flagged: bool,
        crop_x: Optional[int],
        crop_y: Optional[int],
        crop_w: Optional[int],
        crop_h: Optional[int],
        scale_factor: Optional[float],
        centerline_x_px: Optional[float],
        centerline_y_px: Optional[float],
        centerline_x: Optional[float],
        centerline_y: Optional[float],
        centerline_z: Optional[float],
        centroid_x_px: Optional[float],
        centroid_y_px: Optional[float],
        sample_center_x_px: Optional[float],
        sample_center_y_px: Optional[float],
        mask_x: Optional[int],
        mask_y: Optional[int],
        mask_w: Optional[int],
        mask_h: Optional[int],
        full_width: Optional[int],
        full_height: Optional[int],
        raw_json: Optional[Dict[str, Any]],
    ) -> int:
        row = self._fetchone(
            """
            INSERT INTO segment_samples (
                neuron_id,
                segment_id,
                node_db_id,
                run_db_id,
                source_mask_id,
                run_id,
                mask_id,
                frame,
                step,
                image_path,
                mask_path,
                has_mask,
                flagged,
                crop_x,
                crop_y,
                crop_w,
                crop_h,
                scale_factor,
                centerline_x_px,
                centerline_y_px,
                centerline_x,
                centerline_y,
                centerline_z,
                centroid_x_px,
                centroid_y_px,
                sample_center_x_px,
                sample_center_y_px,
                mask_x,
                mask_y,
                mask_w,
                mask_h,
                full_width,
                full_height,
                raw_json
            )
            VALUES (
                %(neuron_id)s,
                %(segment_id)s,
                %(node_db_id)s,
                %(run_db_id)s,
                %(source_mask_id)s,
                %(run_id)s,
                %(mask_id)s,
                %(frame)s,
                %(step)s,
                %(image_path)s,
                %(mask_path)s,
                %(has_mask)s,
                %(flagged)s,
                %(crop_x)s,
                %(crop_y)s,
                %(crop_w)s,
                %(crop_h)s,
                %(scale_factor)s,
                %(centerline_x_px)s,
                %(centerline_y_px)s,
                %(centerline_x)s,
                %(centerline_y)s,
                %(centerline_z)s,
                %(centroid_x_px)s,
                %(centroid_y_px)s,
                %(sample_center_x_px)s,
                %(sample_center_y_px)s,
                %(mask_x)s,
                %(mask_y)s,
                %(mask_w)s,
                %(mask_h)s,
                %(full_width)s,
                %(full_height)s,
                %(raw_json)s
            )
            ON CONFLICT (segment_id, frame, step) DO UPDATE
            SET node_db_id = COALESCE(EXCLUDED.node_db_id, segment_samples.node_db_id),
                run_db_id = COALESCE(EXCLUDED.run_db_id, segment_samples.run_db_id),
                source_mask_id = COALESCE(EXCLUDED.source_mask_id, segment_samples.source_mask_id),
                run_id = COALESCE(EXCLUDED.run_id, segment_samples.run_id),
                mask_id = COALESCE(EXCLUDED.mask_id, segment_samples.mask_id),
                image_path = EXCLUDED.image_path,
                mask_path = COALESCE(EXCLUDED.mask_path, segment_samples.mask_path),
                has_mask = EXCLUDED.has_mask,
                flagged = segment_samples.flagged OR EXCLUDED.flagged,
                crop_x = COALESCE(EXCLUDED.crop_x, segment_samples.crop_x),
                crop_y = COALESCE(EXCLUDED.crop_y, segment_samples.crop_y),
                crop_w = COALESCE(EXCLUDED.crop_w, segment_samples.crop_w),
                crop_h = COALESCE(EXCLUDED.crop_h, segment_samples.crop_h),
                scale_factor = COALESCE(EXCLUDED.scale_factor, segment_samples.scale_factor),
                centerline_x_px = COALESCE(EXCLUDED.centerline_x_px, segment_samples.centerline_x_px),
                centerline_y_px = COALESCE(EXCLUDED.centerline_y_px, segment_samples.centerline_y_px),
                centerline_x = COALESCE(EXCLUDED.centerline_x, segment_samples.centerline_x),
                centerline_y = COALESCE(EXCLUDED.centerline_y, segment_samples.centerline_y),
                centerline_z = COALESCE(EXCLUDED.centerline_z, segment_samples.centerline_z),
                centroid_x_px = COALESCE(EXCLUDED.centroid_x_px, segment_samples.centroid_x_px),
                centroid_y_px = COALESCE(EXCLUDED.centroid_y_px, segment_samples.centroid_y_px),
                sample_center_x_px = COALESCE(EXCLUDED.sample_center_x_px, segment_samples.sample_center_x_px),
                sample_center_y_px = COALESCE(EXCLUDED.sample_center_y_px, segment_samples.sample_center_y_px),
                mask_x = COALESCE(EXCLUDED.mask_x, segment_samples.mask_x),
                mask_y = COALESCE(EXCLUDED.mask_y, segment_samples.mask_y),
                mask_w = COALESCE(EXCLUDED.mask_w, segment_samples.mask_w),
                mask_h = COALESCE(EXCLUDED.mask_h, segment_samples.mask_h),
                full_width = COALESCE(EXCLUDED.full_width, segment_samples.full_width),
                full_height = COALESCE(EXCLUDED.full_height, segment_samples.full_height),
                raw_json = COALESCE(EXCLUDED.raw_json, segment_samples.raw_json)
            RETURNING id
            """,
            {
                "neuron_id": neuron_id,
                "segment_id": segment_id,
                "node_db_id": node_db_id,
                "run_db_id": run_db_id,
                "source_mask_id": source_mask_id,
                "run_id": run_id,
                "mask_id": mask_id,
                "frame": int(frame),
                "step": int(step),
                "image_path": image_path,
                "mask_path": mask_path,
                "has_mask": bool(has_mask),
                "flagged": bool(flagged),
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "scale_factor": scale_factor,
                "centerline_x_px": centerline_x_px,
                "centerline_y_px": centerline_y_px,
                "centerline_x": centerline_x,
                "centerline_y": centerline_y,
                "centerline_z": centerline_z,
                "centroid_x_px": centroid_x_px,
                "centroid_y_px": centroid_y_px,
                "sample_center_x_px": sample_center_x_px,
                "sample_center_y_px": sample_center_y_px,
                "mask_x": mask_x,
                "mask_y": mask_y,
                "mask_w": mask_w,
                "mask_h": mask_h,
                "full_width": full_width,
                "full_height": full_height,
                "raw_json": _jsonb(raw_json),
            },
        )
        assert row is not None
        return int(row["id"])

    def _upsert_reviewer(self, *, username: str, display_name: Optional[str] = None) -> int:
        row = self._fetchone(
            """
            INSERT INTO reviewers (username, display_name)
            VALUES (%(username)s, %(display_name)s)
            ON CONFLICT (username) DO UPDATE
            SET display_name = COALESCE(EXCLUDED.display_name, reviewers.display_name)
            RETURNING id
            """,
            {"username": username, "display_name": display_name},
        )
        assert row is not None
        return int(row["id"])

    def _upsert_assignment(
        self,
        *,
        sample_id: int,
        reviewer_id: int,
        source: str = "import",
        assigned_at: Optional[datetime] = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO review_assignments (sample_id, reviewer_id, assigned_at, source)
            VALUES (%(sample_id)s, %(reviewer_id)s, %(assigned_at)s, %(source)s)
            ON CONFLICT (sample_id, reviewer_id) DO UPDATE
            SET source = EXCLUDED.source
            """,
            {
                "sample_id": sample_id,
                "reviewer_id": reviewer_id,
                "assigned_at": assigned_at or _utc_now(),
                "source": source,
            },
        )

    def _upsert_review(
        self,
        *,
        sample_id: int,
        reviewer_id: int,
        verdict: str,
        delta_mask_path: Optional[str],
        delta_mask_png: Optional[bytes],
        submitted_at: Optional[datetime],
        raw_json: Optional[Dict[str, Any]],
    ) -> None:
        self._execute(
            """
            INSERT INTO reviews (
                sample_id,
                reviewer_id,
                verdict,
                delta_mask_path,
                delta_mask_png,
                submitted_at,
                raw_json
            )
            VALUES (
                %(sample_id)s,
                %(reviewer_id)s,
                %(verdict)s,
                %(delta_mask_path)s,
                %(delta_mask_png)s,
                %(submitted_at)s,
                %(raw_json)s
            )
            ON CONFLICT (sample_id, reviewer_id) DO UPDATE
            SET verdict = EXCLUDED.verdict,
                delta_mask_path = COALESCE(EXCLUDED.delta_mask_path, reviews.delta_mask_path),
                delta_mask_png = COALESCE(EXCLUDED.delta_mask_png, reviews.delta_mask_png),
                submitted_at = EXCLUDED.submitted_at,
                raw_json = COALESCE(EXCLUDED.raw_json, reviews.raw_json)
            """,
            {
                "sample_id": sample_id,
                "reviewer_id": reviewer_id,
                "verdict": verdict,
                "delta_mask_path": delta_mask_path,
                "delta_mask_png": delta_mask_png,
                "submitted_at": submitted_at or _utc_now(),
                "raw_json": _jsonb(raw_json),
            },
        )

    def import_neurotracer_neuron(
        self,
        *,
        data_root: Path,
        neuron_id: str,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized = _normalize_neuron_id(neuron_id)
        neuron_name = f"neuron_{normalized}"
        store_dir = Path(data_root) / neuron_name / "masks"
        index_path = store_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Mask index not found: {index_path}")
        with index_path.open("r", encoding="utf-8") as handle:
            index_rows = json.load(handle)
        if not isinstance(index_rows, list):
            raise ValueError(f"Invalid mask index format: {index_path}")

        resolved_rows: List[Dict[str, Any]] = []
        run_summary: Dict[str, Dict[str, Any]] = {}
        for entry in index_rows:
            run_id = str(entry.get("run_id") or "").strip()
            if not run_id:
                continue
            rel_path = str(entry.get("path") or "").strip()
            mask_path = store_dir / rel_path if rel_path else None
            width = _coerce_int(entry.get("width"))
            height = _coerce_int(entry.get("height"))
            if width <= 0 or height <= 0:
                width, height = _mask_dimensions(mask_path)
            color = _parse_color(entry.get("color"))
            run_item = run_summary.setdefault(
                run_id,
                {
                    "direction": str(entry.get("direction") or "").strip() or None,
                    "run_started": str(entry.get("run_started") or "").strip() or None,
                    "color": color,
                    "max_width": 0,
                    "max_height": 0,
                },
            )
            run_item["max_width"] = max(int(run_item["max_width"]), int(width))
            run_item["max_height"] = max(int(run_item["max_height"]), int(height))
            if run_item.get("color") is None and color is not None:
                run_item["color"] = color
            mask_id = str(entry.get("id") or "").strip() or (mask_path.stem if mask_path else f"{run_id}_{_coerce_int(entry.get('frame')):05d}")
            resolved_rows.append(
                {
                    "run_id": run_id,
                    "mask_id": mask_id,
                    "frame": _coerce_int(entry.get("frame")),
                    "x": _coerce_int(entry.get("x")),
                    "y": _coerce_int(entry.get("y")),
                    "width": width,
                    "height": height,
                    "full_width": _coerce_int(entry.get("full_width")),
                    "full_height": _coerce_int(entry.get("full_height")),
                    "path": None if mask_path is None else str(mask_path),
                    "flagged": False,
                    "color": color,
                    "raw_json": entry,
                }
            )

        dataset_name = dataset_name or Path(data_root).name
        run_ids: Dict[str, int] = {}
        try:
            dataset_id = self._upsert_dataset(name=dataset_name, source_root=str(Path(data_root)))
            neuron_db_id = self._upsert_neuron(
                dataset_id=dataset_id,
                name=neuron_name,
                external_id=normalized,
                source_type="neurotracer",
                export_root=None,
                pixel_size_xy=None,
                slice_thickness_z=None,
                raw_json={"index_path": str(index_path)},
            )
            for run_id, info in run_summary.items():
                run_ids[run_id] = self._upsert_run(
                    neuron_id=neuron_db_id,
                    run_id=run_id,
                    direction=info.get("direction"),
                    run_started=info.get("run_started"),
                    color=info.get("color"),
                    max_width=_coerce_int(info.get("max_width")),
                    max_height=_coerce_int(info.get("max_height")),
                    raw_json=None,
                )
            for row in resolved_rows:
                self._upsert_source_mask(
                    neuron_id=neuron_db_id,
                    run_db_id=run_ids.get(row["run_id"]),
                    mask_id=row["mask_id"],
                    frame=row["frame"],
                    x=row["x"],
                    y=row["y"],
                    width=row["width"],
                    height=row["height"],
                    full_width=row["full_width"],
                    full_height=row["full_height"],
                    path=row["path"],
                    color=row["color"],
                    flagged=False,
                    raw_json=row["raw_json"],
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return {
            "dataset": dataset_name,
            "neuron": neuron_name,
            "runs": len(run_ids),
            "source_masks": len(resolved_rows),
        }

    def import_export_bundle(
        self,
        *,
        export_root: Path,
        dataset_name: Optional[str] = None,
        include_reviews: bool = True,
    ) -> Dict[str, Any]:
        export_root = Path(export_root)
        sample_rows = self._load_export_sample_rows(export_root)
        if not sample_rows:
            raise FileNotFoundError(f"No sample_points data found in {export_root}")
        neuron_name = str(sample_rows[0].get("neuron") or export_root.name)
        skeleton_data = self._load_skeleton(export_root / "skeleton.json")
        spacing = skeleton_data.get("spacing", {})
        metadata_map = self._load_metadata_rows(export_root / "metadata.csv")
        flags_rows = self._load_flags_rows(export_root / "flags.csv")
        segment_defs = self._load_segment_definitions(
            export_root / "segment_tree.json",
            sample_rows,
            skeleton_data.get("segments", {}),
            skeleton_data.get("nodes", []),
        )

        resolved_flag_rows: List[Dict[str, Any]] = []
        run_summary: Dict[str, Dict[str, Any]] = {}
        for entry in flags_rows:
            run_id = str(entry.get("run_id") or "").strip()
            if not run_id:
                continue
            path_value = str(entry.get("path") or "").strip()
            npz_path = Path(path_value) if path_value else None
            width = _coerce_int(entry.get("width"))
            height = _coerce_int(entry.get("height"))
            if width <= 0 or height <= 0:
                width, height = _mask_dimensions(npz_path)
            run_item = run_summary.setdefault(
                run_id,
                {"direction": None, "run_started": None, "color": None, "max_width": 0, "max_height": 0},
            )
            run_item["max_width"] = max(int(run_item["max_width"]), int(width))
            run_item["max_height"] = max(int(run_item["max_height"]), int(height))
            resolved_flag_rows.append(
                {
                    "run_id": run_id,
                    "mask_id": str(entry.get("mask_id") or "").strip(),
                    "frame": _coerce_int(entry.get("frame")),
                    "x": _coerce_int(entry.get("x")),
                    "y": _coerce_int(entry.get("y")),
                    "width": width,
                    "height": height,
                    "full_width": _coerce_int(entry.get("full_width")),
                    "full_height": _coerce_int(entry.get("full_height")),
                    "path": path_value or None,
                    "flagged": _coerce_bool(entry.get("flagged")),
                    "color": None,
                    "raw_json": entry,
                }
            )

        dataset_name = dataset_name or export_root.name
        run_ids: Dict[str, int] = {}
        source_mask_ids: Dict[str, int] = {}
        segment_ids: Dict[int, int] = {}
        node_db_ids: Dict[int, int] = {}
        sample_key_to_id: Dict[Tuple[str, int, int], int] = {}
        review_count = 0
        assignments: List[Dict[str, Any]] = []

        try:
            dataset_id = self._upsert_dataset(name=dataset_name, source_root=str(export_root))
            neuron_db_id = self._upsert_neuron(
                dataset_id=dataset_id,
                name=neuron_name,
                external_id=_normalize_neuron_id(neuron_name),
                source_type="export",
                export_root=str(export_root),
                pixel_size_xy=_coerce_float(spacing.get("pixel_size_xy")),
                slice_thickness_z=_coerce_float(spacing.get("slice_thickness_z")),
                raw_json={"export_root": str(export_root)},
            )

            for run_id, info in run_summary.items():
                run_ids[run_id] = self._upsert_run(
                    neuron_id=neuron_db_id,
                    run_id=run_id,
                    direction=info.get("direction"),
                    run_started=info.get("run_started"),
                    color=info.get("color"),
                    max_width=_coerce_int(info.get("max_width")),
                    max_height=_coerce_int(info.get("max_height")),
                    raw_json=None,
                )

            for row in resolved_flag_rows:
                if not row["mask_id"]:
                    continue
                source_mask_ids[row["mask_id"]] = self._upsert_source_mask(
                    neuron_id=neuron_db_id,
                    run_db_id=run_ids.get(row["run_id"]),
                    mask_id=row["mask_id"],
                    frame=row["frame"],
                    x=row["x"],
                    y=row["y"],
                    width=row["width"],
                    height=row["height"],
                    full_width=row["full_width"],
                    full_height=row["full_height"],
                    path=row["path"],
                    color=row["color"],
                    flagged=row["flagged"],
                    raw_json=row["raw_json"],
                )

            for segment_def in segment_defs:
                segment_ids[int(segment_def["segment_index"])] = self._upsert_segment(
                    neuron_id=neuron_db_id,
                    segment_index=int(segment_def["segment_index"]),
                    segment_path=str(segment_def["segment_path"]),
                    parent_segment_index=segment_def.get("parent_segment_index"),
                    node_count=_coerce_int(segment_def.get("node_count")),
                    frame_count=_coerce_int(segment_def.get("frame_count")),
                    raw_json=segment_def.get("raw_json"),
                )

            for node in skeleton_data.get("nodes", []):
                node_db_ids[int(node["node_index"])] = self._upsert_skeleton_node(
                    neuron_id=neuron_db_id,
                    node_index=int(node["node_index"]),
                    frame=_coerce_int(node.get("frame")),
                    x=_coerce_float(node.get("x")),
                    y=_coerce_float(node.get("y")),
                    z=_coerce_float(node.get("z")),
                    x_px=_coerce_float(node.get("x_px")),
                    y_px=_coerce_float(node.get("y_px")),
                    label=node.get("label"),
                    degree=_coerce_int(node.get("degree")) if node.get("degree") is not None else None,
                    run_id=node.get("run_id"),
                    raw_json=node.get("raw_json"),
                )

            for edge in skeleton_data.get("edges", []):
                left = node_db_ids.get(int(edge[0]))
                right = node_db_ids.get(int(edge[1]))
                if left is None or right is None:
                    continue
                self._insert_skeleton_edge(
                    neuron_id=neuron_db_id,
                    source_node_db_id=left,
                    target_node_db_id=right,
                )

            for seg_index, node_indexes in skeleton_data.get("segments", {}).items():
                segment_db_id = segment_ids.get(int(seg_index))
                if segment_db_id is None:
                    continue
                for node_order, node_index in enumerate(node_indexes):
                    node_db_id = node_db_ids.get(int(node_index))
                    if node_db_id is None:
                        continue
                    self._insert_segment_node(
                        segment_id=segment_db_id,
                        node_db_id=node_db_id,
                        node_order=node_order,
                    )

            for row in sample_rows:
                segment_path = str(row.get("segment") or "")
                frame = _coerce_int(row.get("frame"))
                step = _coerce_int(row.get("step"))
                metadata_row = metadata_map.get((segment_path, frame, step), {})
                segment_index = _coerce_int(row.get("segment_index"), default=_coerce_int(metadata_row.get("segment_index")))
                segment_db_id = segment_ids.get(segment_index)
                if segment_db_id is None:
                    segment_db_id = self._upsert_segment(
                        neuron_id=neuron_db_id,
                        segment_index=segment_index,
                        segment_path=segment_path,
                        parent_segment_index=None,
                        node_count=0,
                        frame_count=0,
                        raw_json={"segment": segment_path},
                    )
                    segment_ids[segment_index] = segment_db_id
                run_id = str(row.get("run_id") or metadata_row.get("run_id") or "").strip() or None
                mask_id = str(row.get("mask_id") or metadata_row.get("mask_id") or "").strip() or None
                sample_id = self._upsert_segment_sample(
                    neuron_id=neuron_db_id,
                    segment_id=segment_db_id,
                    node_db_id=node_db_ids.get(_coerce_int(row.get("node_id"))) if row.get("node_id") not in (None, "") else None,
                    run_db_id=None if run_id is None else run_ids.get(run_id),
                    source_mask_id=None if mask_id is None else source_mask_ids.get(mask_id),
                    run_id=run_id,
                    mask_id=mask_id,
                    frame=frame,
                    step=step,
                    image_path=str(row.get("image_path") or metadata_row.get("image_path") or ""),
                    mask_path=str(row.get("mask_path") or metadata_row.get("mask_path") or "") or None,
                    has_mask=_coerce_bool(row.get("has_mask") if "has_mask" in row else metadata_row.get("mask_path")),
                    flagged=_coerce_bool(row.get("flagged") if "flagged" in row else metadata_row.get("flagged")),
                    crop_x=_coerce_int(row.get("crop_x")) if row.get("crop_x") not in (None, "") else None,
                    crop_y=_coerce_int(row.get("crop_y")) if row.get("crop_y") not in (None, "") else None,
                    crop_w=_coerce_int(row.get("crop_w")) if row.get("crop_w") not in (None, "") else None,
                    crop_h=_coerce_int(row.get("crop_h")) if row.get("crop_h") not in (None, "") else None,
                    scale_factor=_coerce_float(row.get("scale_factor")),
                    centerline_x_px=_coerce_float(row.get("centerline_x_px")),
                    centerline_y_px=_coerce_float(row.get("centerline_y_px")),
                    centerline_x=_coerce_float(row.get("centerline_x")),
                    centerline_y=_coerce_float(row.get("centerline_y")),
                    centerline_z=_coerce_float(row.get("centerline_z")),
                    centroid_x_px=_coerce_float(row.get("centroid_x_px")),
                    centroid_y_px=_coerce_float(row.get("centroid_y_px")),
                    sample_center_x_px=_coerce_float(row.get("sample_center_x_px")),
                    sample_center_y_px=_coerce_float(row.get("sample_center_y_px")),
                    mask_x=_coerce_int(metadata_row.get("mask_x")) if metadata_row.get("mask_x") not in (None, "") else None,
                    mask_y=_coerce_int(metadata_row.get("mask_y")) if metadata_row.get("mask_y") not in (None, "") else None,
                    mask_w=_coerce_int(metadata_row.get("mask_w")) if metadata_row.get("mask_w") not in (None, "") else None,
                    mask_h=_coerce_int(metadata_row.get("mask_h")) if metadata_row.get("mask_h") not in (None, "") else None,
                    full_width=_coerce_int(metadata_row.get("full_width")) if metadata_row.get("full_width") not in (None, "") else None,
                    full_height=_coerce_int(metadata_row.get("full_height")) if metadata_row.get("full_height") not in (None, "") else None,
                    raw_json=row,
                )
                sample_key_to_id[(segment_path, frame, step)] = sample_id

            assignments = self._load_assignments(export_root / "assignments.csv")
            for assignment in assignments:
                sample_id = sample_key_to_id.get(
                    (assignment["segment"], assignment["frame"], assignment["step"])
                )
                if sample_id is None:
                    continue
                reviewer_id = self._upsert_reviewer(username=assignment["reviewer"])
                self._upsert_assignment(
                    sample_id=sample_id,
                    reviewer_id=reviewer_id,
                    source="assignments.csv",
                )

            if include_reviews:
                for review in self._load_reviews(export_root):
                    sample_id = sample_key_to_id.get(
                        (review["segment"], review["frame"], review["step"])
                    )
                    if sample_id is None:
                        continue
                    reviewer_id = self._upsert_reviewer(username=review["reviewer"])
                    self._upsert_assignment(
                        sample_id=sample_id,
                        reviewer_id=reviewer_id,
                        source="reviews",
                    )
                    self._upsert_review(
                        sample_id=sample_id,
                        reviewer_id=reviewer_id,
                        verdict=review["verdict"],
                        delta_mask_path=review.get("delta_mask_path"),
                        delta_mask_png=review.get("delta_mask_png"),
                        submitted_at=_utc_now(),
                        raw_json={
                            "segment": review["segment"],
                            "frame": review["frame"],
                            "step": review["step"],
                            "reviewer": review["reviewer"],
                            "verdict": review["verdict"],
                        },
                    )
                    review_count += 1
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return {
            "dataset": dataset_name,
            "neuron": neuron_name,
            "segments": len(segment_ids),
            "nodes": len(node_db_ids),
            "source_masks": len(source_mask_ids),
            "samples": len(sample_rows),
            "assignments": len(assignments),
            "reviews": review_count if include_reviews else 0,
        }

    def record_review(
        self,
        *,
        neuron_name: str,
        segment_path: str,
        frame: int,
        step: int,
        reviewer: str,
        verdict: str,
        delta_mask: Optional[np.ndarray] = None,
        delta_mask_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        verdict = str(verdict).strip().lower()
        if verdict not in {"good", "bad", "pending"}:
            raise ValueError(f"Unsupported verdict: {verdict}")
        sample = self._fetchone(
            """
            SELECT ss.id AS sample_id
            FROM segment_samples ss
            JOIN segments s ON s.id = ss.segment_id
            JOIN neurons n ON n.id = ss.neuron_id
            WHERE n.name = %(neuron_name)s
              AND s.segment_path = %(segment_path)s
              AND ss.frame = %(frame)s
              AND ss.step = %(step)s
            """,
            {
                "neuron_name": neuron_name,
                "segment_path": segment_path,
                "frame": int(frame),
                "step": int(step),
            },
        )
        if sample is None:
            raise KeyError(
                f"No segment sample found for neuron={neuron_name} segment={segment_path} frame={frame} step={step}"
            )
        reviewer_id = self._upsert_reviewer(username=reviewer)
        delta_png = _mask_png_bytes(delta_mask) if delta_mask is not None else None
        path_value = None if delta_mask_path is None else str(delta_mask_path)
        if delta_png is None and delta_mask_path is not None and Path(delta_mask_path).exists():
            delta_png = Path(delta_mask_path).read_bytes()
        try:
            self._upsert_assignment(
                sample_id=int(sample["sample_id"]),
                reviewer_id=reviewer_id,
                source="record_review",
            )
            self._upsert_review(
                sample_id=int(sample["sample_id"]),
                reviewer_id=reviewer_id,
                verdict=verdict,
                delta_mask_path=path_value,
                delta_mask_png=delta_png,
                submitted_at=_utc_now(),
                raw_json=metadata,
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_segment_samples(
        self,
        *,
        neuron_name: str,
        segment_path: str,
        frame: Optional[int] = None,
        step: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        conditions = [
            "n.name = %(neuron_name)s",
            "s.segment_path = %(segment_path)s",
        ]
        params: Dict[str, Any] = {
            "neuron_name": neuron_name,
            "segment_path": segment_path,
        }
        if frame is not None:
            conditions.append("ss.frame = %(frame)s")
            params["frame"] = int(frame)
        if step is not None:
            conditions.append("ss.step = %(step)s")
            params["step"] = int(step)
        query = f"""
            SELECT
                ss.id AS sample_id,
                n.name AS neuron,
                s.segment_path AS segment,
                s.segment_index,
                ss.frame,
                ss.step,
                ss.image_path,
                ss.mask_path,
                ss.has_mask,
                ss.flagged,
                ss.run_id,
                ss.mask_id,
                ss.crop_x,
                ss.crop_y,
                ss.crop_w,
                ss.crop_h,
                ss.scale_factor,
                ss.centerline_x_px,
                ss.centerline_y_px,
                ss.centerline_x,
                ss.centerline_y,
                ss.centerline_z,
                ss.centroid_x_px,
                ss.centroid_y_px,
                ss.sample_center_x_px,
                ss.sample_center_y_px,
                ss.mask_x,
                ss.mask_y,
                ss.mask_w,
                ss.mask_h,
                ss.full_width,
                ss.full_height
            FROM segment_samples ss
            JOIN segments s ON s.id = ss.segment_id
            JOIN neurons n ON n.id = ss.neuron_id
            WHERE {" AND ".join(conditions)}
            ORDER BY ss.step, ss.frame
        """
        return self._fetchall(query, params)

    def segment_consensus_rows(
        self,
        *,
        neuron_name: str,
        segment_path: str,
        frame: Optional[int] = None,
        step: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        samples = self.get_segment_samples(
            neuron_name=neuron_name,
            segment_path=segment_path,
            frame=frame,
            step=step,
        )
        if not samples:
            return []

        conditions = [
            "n.name = %(neuron_name)s",
            "s.segment_path = %(segment_path)s",
        ]
        params: Dict[str, Any] = {
            "neuron_name": neuron_name,
            "segment_path": segment_path,
        }
        if frame is not None:
            conditions.append("ss.frame = %(frame)s")
            params["frame"] = int(frame)
        if step is not None:
            conditions.append("ss.step = %(step)s")
            params["step"] = int(step)
        where_clause = " AND ".join(conditions)

        assignment_rows = self._fetchall(
            f"""
            SELECT ss.id AS sample_id, COUNT(ra.id) AS assignment_count
            FROM segment_samples ss
            JOIN segments s ON s.id = ss.segment_id
            JOIN neurons n ON n.id = ss.neuron_id
            LEFT JOIN review_assignments ra ON ra.sample_id = ss.id
            WHERE {where_clause}
            GROUP BY ss.id
            """,
            params,
        )
        assignment_counts = {
            int(row["sample_id"]): int(row["assignment_count"]) for row in assignment_rows
        }

        review_rows = self._fetchall(
            f"""
            SELECT
                ss.id AS sample_id,
                rv.verdict,
                rv.delta_mask_path,
                rv.delta_mask_png
            FROM segment_samples ss
            JOIN segments s ON s.id = ss.segment_id
            JOIN neurons n ON n.id = ss.neuron_id
            LEFT JOIN reviews rv ON rv.sample_id = ss.id
            WHERE {where_clause}
            ORDER BY ss.id
            """,
            params,
        )

        reviews_by_sample: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for review in review_rows:
            if review.get("verdict") is None:
                continue
            reviews_by_sample[int(review["sample_id"])].append(review)

        result: List[Dict[str, Any]] = []
        for sample in samples:
            sample_id = int(sample["sample_id"])
            sample_reviews = reviews_by_sample.get(sample_id, [])
            num_good = sum(1 for item in sample_reviews if item["verdict"] == "good")
            num_bad = sum(1 for item in sample_reviews if item["verdict"] == "bad")
            completed = num_good + num_bad
            num_reviewers = max(int(assignment_counts.get(sample_id, 0)), len(sample_reviews))
            num_pending = max(num_reviewers - completed, 0)

            bad_masks: List[np.ndarray] = []
            for item in sample_reviews:
                if item["verdict"] != "bad":
                    continue
                mask = None
                if item.get("delta_mask_png") is not None:
                    mask = _load_mask_from_png_bytes(item["delta_mask_png"])
                if mask is None:
                    mask = _load_mask_from_png_path(item.get("delta_mask_path"))
                if mask is not None:
                    bad_masks.append(mask)

            mask_agreement = None
            if len(bad_masks) >= 2:
                ious: List[float] = []
                for i in range(len(bad_masks)):
                    for j in range(i + 1, len(bad_masks)):
                        ious.append(mask_iou(bad_masks[i], bad_masks[j]))
                mask_agreement = _mean(ious)

            scores = score_bundle(
                num_good=num_good,
                num_bad=num_bad,
                num_pending=num_pending,
                mask_agreement=mask_agreement,
            )
            row = dict(sample)
            row.update(
                {
                    "num_reviewers": num_reviewers,
                    "num_good": num_good,
                    "num_bad": num_bad,
                    "num_pending": num_pending,
                    "mask_agreement": mask_agreement,
                    **scores,
                }
            )
            result.append(row)
        return result

    def segment_consensus_summary(
        self,
        *,
        neuron_name: str,
        segment_path: str,
        frame: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        rows = self.segment_consensus_rows(
            neuron_name=neuron_name,
            segment_path=segment_path,
            frame=frame,
            step=step,
        )
        return {
            "neuron": neuron_name,
            "segment": segment_path,
            "sample_count": len(rows),
            "reviewed_sample_count": sum(1 for row in rows if row["num_good"] + row["num_bad"] > 0),
            "total_reviewers": sum(int(row["num_reviewers"]) for row in rows),
            "total_good": sum(int(row["num_good"]) for row in rows),
            "total_bad": sum(int(row["num_bad"]) for row in rows),
            "total_pending": sum(int(row["num_pending"]) for row in rows),
            "avg_acceptance_ratio": _mean(row.get("acceptance_ratio") for row in rows),
            "avg_agreement_ratio": _mean(row.get("agreement_ratio") for row in rows),
            "avg_hybrid_consensus": _mean(row.get("hybrid_consensus") for row in rows),
            "avg_mask_agreement": _mean(row.get("mask_agreement") for row in rows),
            "consensus_score": _mean(row.get("consensus_score") for row in rows),
        }

    def compare_segment_scoring(
        self,
        *,
        neuron_name: str,
        segment_path: str,
        frame: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "summary": self.segment_consensus_summary(
                neuron_name=neuron_name,
                segment_path=segment_path,
                frame=frame,
                step=step,
            ),
            "samples": self.segment_consensus_rows(
                neuron_name=neuron_name,
                segment_path=segment_path,
                frame=frame,
                step=step,
            ),
        }

    def _load_export_sample_rows(self, export_root: Path) -> List[Dict[str, Any]]:
        json_path = export_root / "sample_points.json"
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return [dict(item) for item in payload]
        csv_path = export_root / "sample_points.csv"
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        return []

    def _load_metadata_rows(self, path: Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        return {
            (str(row.get("segment") or ""), _coerce_int(row.get("frame")), _coerce_int(row.get("step"))): row
            for row in rows
        }

    def _load_flags_rows(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _load_assignments(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        return [
            {
                "segment": str(row.get("segment") or ""),
                "frame": _coerce_int(row.get("frame")),
                "step": _coerce_int(row.get("step")),
                "reviewer": str(row.get("reviewer") or "").strip(),
            }
            for row in rows
            if str(row.get("reviewer") or "").strip()
        ]

    def _load_reviews(self, export_root: Path) -> List[Dict[str, Any]]:
        reviews_dir = export_root / "reviews"
        if not reviews_dir.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for user_dir in sorted(reviews_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            verdicts_path = user_dir / "verdicts.json"
            if not verdicts_path.exists():
                continue
            with verdicts_path.open("r", encoding="utf-8") as handle:
                verdicts = json.load(handle)
            if not isinstance(verdicts, dict):
                continue
            for frame_key, verdict in verdicts.items():
                segment, frame, step = _parse_frame_key(frame_key)
                delta_path = export_root / "reviews" / user_dir.name / "deltas" / Path(segment) / f"frame_{frame:05d}_mask.png"
                rows.append(
                    {
                        "reviewer": user_dir.name,
                        "segment": segment,
                        "frame": frame,
                        "step": step,
                        "verdict": str(verdict).strip().lower(),
                        "delta_mask_path": str(delta_path) if delta_path.exists() else None,
                        "delta_mask_png": delta_path.read_bytes() if delta_path.exists() else None,
                    }
                )
        return rows

    def _load_segment_definitions(
        self,
        segment_tree_path: Path,
        sample_rows: Sequence[Dict[str, Any]],
        skeleton_segments: Dict[int, List[int]],
        skeleton_nodes: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if segment_tree_path.exists():
            with segment_tree_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            result = []
            for key, value in sorted((payload.get("segments") or {}).items(), key=lambda item: int(item[0])):
                result.append(
                    {
                        "segment_index": int(key),
                        "segment_path": str(value.get("path") or f"segment_{key}"),
                        "parent_segment_index": value.get("parent"),
                        "node_count": _coerce_int(value.get("node_count")),
                        "frame_count": _coerce_int(value.get("frame_count")),
                        "raw_json": value,
                    }
                )
            return result

        by_segment: Dict[int, Dict[str, Any]] = {}
        frame_sets: Dict[int, set] = defaultdict(set)
        for row in sample_rows:
            seg_index = _coerce_int(row.get("segment_index"))
            item = by_segment.setdefault(
                seg_index,
                {
                    "segment_index": seg_index,
                    "segment_path": str(row.get("segment") or f"segment_{seg_index}"),
                    "parent_segment_index": None,
                    "node_count": 0,
                    "frame_count": 0,
                    "raw_json": {"derived_from": "sample_points"},
                },
            )
            frame_sets[seg_index].add(_coerce_int(row.get("frame")))
            item["node_count"] = max(int(item["node_count"]), len(skeleton_segments.get(seg_index, [])))
        for seg_index, item in by_segment.items():
            item["frame_count"] = len(frame_sets.get(seg_index, set()))
            if item["node_count"] <= 0 and skeleton_segments.get(seg_index):
                item["node_count"] = len(skeleton_segments[seg_index])
        if by_segment:
            return [by_segment[idx] for idx in sorted(by_segment)]

        result = []
        for seg_index, node_indexes in sorted(skeleton_segments.items()):
            frames = {
                _coerce_int(skeleton_nodes[node_index].get("frame"))
                for node_index in node_indexes
                if 0 <= int(node_index) < len(skeleton_nodes)
            }
            result.append(
                {
                    "segment_index": int(seg_index),
                    "segment_path": f"segment_{seg_index}",
                    "parent_segment_index": None,
                    "node_count": len(node_indexes),
                    "frame_count": len(frames),
                    "raw_json": {"derived_from": "skeleton"},
                }
            )
        return result

    def _load_skeleton(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"nodes": [], "edges": [], "segments": {}, "spacing": {}}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        edges = payload.get("edges") or []
        spacing = {
            "pixel_size_xy": payload.get("pixel_size_xy"),
            "slice_thickness_z": payload.get("slice_thickness_z"),
        }
        if "spacing" in payload and isinstance(payload["spacing"], dict):
            spacing["pixel_size_xy"] = payload["spacing"].get("pixel_size_xy", spacing["pixel_size_xy"])
            spacing["slice_thickness_z"] = payload["spacing"].get("slice_thickness_z", spacing["slice_thickness_z"])

        nodes_data = payload.get("nodes") or []
        parsed_nodes: List[Dict[str, Any]] = []
        if nodes_data and isinstance(nodes_data[0], dict):
            for node in nodes_data:
                parsed_nodes.append(
                    {
                        "node_index": _coerce_int(node.get("id")),
                        "frame": _coerce_int(node.get("frame")),
                        "x": _coerce_float(node.get("x")),
                        "y": _coerce_float(node.get("y")),
                        "z": _coerce_float(node.get("z")),
                        "x_px": _coerce_float(node.get("x_px")),
                        "y_px": _coerce_float(node.get("y_px")),
                        "label": node.get("label"),
                        "degree": _coerce_int(node.get("degree")) if node.get("degree") is not None else None,
                        "run_id": node.get("run_id"),
                        "raw_json": node,
                    }
                )
        else:
            pixel_coords = payload.get("node_pixel_coords") or []
            node_frames = payload.get("node_frames") or []
            degree_counts: Dict[int, int] = defaultdict(int)
            for edge in edges:
                if len(edge) < 2:
                    continue
                degree_counts[int(edge[0])] += 1
                degree_counts[int(edge[1])] += 1
            for idx, coords in enumerate(nodes_data):
                pixel = pixel_coords[idx] if idx < len(pixel_coords) else [None, None]
                frame = node_frames[idx] if idx < len(node_frames) else 0
                parsed_nodes.append(
                    {
                        "node_index": idx,
                        "frame": _coerce_int(frame),
                        "x": _coerce_float(coords[0] if len(coords) > 0 else None),
                        "y": _coerce_float(coords[1] if len(coords) > 1 else None),
                        "z": _coerce_float(coords[2] if len(coords) > 2 else None),
                        "x_px": _coerce_float(pixel[0] if len(pixel) > 0 else None),
                        "y_px": _coerce_float(pixel[1] if len(pixel) > 1 else None),
                        "label": None,
                        "degree": degree_counts.get(idx),
                        "run_id": "mesh",
                        "raw_json": {
                            "node_index": idx,
                            "coords": coords,
                            "pixel_coords": pixel,
                            "frame": frame,
                        },
                    }
                )

        segment_map = {
            int(idx): [int(node_index) for node_index in node_indexes]
            for idx, node_indexes in enumerate(payload.get("segments") or [])
        }
        return {
            "nodes": parsed_nodes,
            "edges": [[int(edge[0]), int(edge[1])] for edge in edges if len(edge) >= 2],
            "segments": segment_map,
            "spacing": spacing,
        }
