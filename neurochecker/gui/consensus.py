"""Deterministic frame assignment and consensus scoring for multi-reviewer workflows."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from neurochecker.consensus_metrics import mask_iou, score_bundle

logger = logging.getLogger("neurochecker")

FrameKey = Tuple[str, int, int]  # (segment, frame, step)


def read_volunteers(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return []
    names: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            if name:
                names.append(name)
    return sorted(set(names))


def read_sample_points(csv_path: Path) -> List[FrameKey]:
    if not csv_path.exists():
        return []
    frames: List[FrameKey] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg = row.get("segment", "")
            frame = int(row.get("frame", 0))
            step = int(row.get("step", 0))
            frames.append((seg, frame, step))
    return frames


def generate_assignments(
    sample_points_csv: Path,
    volunteers_csv: Path,
    reviewers_per_frame: int = 2,
) -> List[Tuple[str, int, int, str]]:
    volunteers = read_volunteers(volunteers_csv)
    if not volunteers:
        return []
    frames = read_sample_points(sample_points_csv)
    if not frames:
        return []
    reviewers_per_frame = min(reviewers_per_frame, len(volunteers))
    assignments: List[Tuple[str, int, int, str]] = []
    for i, (seg, frame, step) in enumerate(frames):
        for k in range(reviewers_per_frame):
            reviewer = volunteers[(i + k) % len(volunteers)]
            assignments.append((seg, frame, step, reviewer))
    return assignments


def write_assignments(assignments: List[Tuple[str, int, int, str]], path: Path) -> None:
    if not assignments:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "frame", "step", "reviewer"])
        for seg, frame, step, reviewer in assignments:
            writer.writerow([seg, frame, step, reviewer])


def read_assignments(path: Path) -> List[Tuple[str, int, int, str]]:
    if not path.exists():
        return []
    result: List[Tuple[str, int, int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append((
                row["segment"],
                int(row["frame"]),
                int(row["step"]),
                row["reviewer"],
            ))
    return result


def assignments_for_user(
    assignments: List[Tuple[str, int, int, str]], username: str
) -> List[FrameKey]:
    return [
        (seg, frame, step)
        for seg, frame, step, reviewer in assignments
        if reviewer == username
    ]


def load_verdicts(export_root: Path, username: str) -> Dict[str, str]:
    path = export_root / "reviews" / username / "verdicts.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_verdicts(export_root: Path, username: str, verdicts: Dict[str, str]) -> None:
    user_dir = export_root / "reviews" / username
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / "verdicts.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(verdicts, f, indent=2)


def frame_key_str(seg: str, frame: int, step: int) -> str:
    return f"{seg}/{frame:05d}/{step:05d}"


def delta_mask_path(export_root: Path, username: str, seg: str, frame: int) -> Path:
    return export_root / "reviews" / username / "deltas" / seg / f"frame_{frame:05d}_mask.png"


def save_delta_mask(export_root: Path, username: str, seg: str, frame: int, mask: np.ndarray) -> Path:
    path = delta_mask_path(export_root, username, seg, frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)
    return path


def load_delta_mask(export_root: Path, username: str, seg: str, frame: int) -> Optional[np.ndarray]:
    path = delta_mask_path(export_root, username, seg, frame)
    if not path.exists():
        return None
    try:
        img = np.array(Image.open(path).convert("L"))
        return (img > 127).astype(np.uint8)
    except Exception:
        return None


def compute_consensus(export_root: Path) -> List[Dict[str, Any]]:
    assignments_path = export_root / "assignments.csv"
    assignments = read_assignments(assignments_path)
    if not assignments:
        return []

    frame_reviewers: Dict[FrameKey, List[str]] = {}
    for seg, frame, step, reviewer in assignments:
        key = (seg, frame, step)
        frame_reviewers.setdefault(key, []).append(reviewer)

    all_verdicts: Dict[str, Dict[str, str]] = {}
    reviews_dir = export_root / "reviews"
    if reviews_dir.exists():
        for user_dir in reviews_dir.iterdir():
            if user_dir.is_dir():
                all_verdicts[user_dir.name] = load_verdicts(export_root, user_dir.name)

    rows: List[Dict[str, Any]] = []
    for (seg, frame, step), reviewers in sorted(frame_reviewers.items()):
        key_s = frame_key_str(seg, frame, step)
        num_good = 0
        num_bad = 0
        num_pending = 0
        delta_masks: List[np.ndarray] = []

        for reviewer in reviewers:
            verdict = all_verdicts.get(reviewer, {}).get(key_s)
            if verdict == "good":
                num_good += 1
            elif verdict == "bad":
                num_bad += 1
                dm = load_delta_mask(export_root, reviewer, seg, frame)
                if dm is not None:
                    delta_masks.append(dm)
            else:
                num_pending += 1

        mask_agreement: Optional[float] = None
        if len(delta_masks) >= 2:
            ious = []
            for i in range(len(delta_masks)):
                for j in range(i + 1, len(delta_masks)):
                    ious.append(mask_iou(delta_masks[i], delta_masks[j]))
            mask_agreement = float(np.mean(ious)) if ious else None

        scores = score_bundle(
            num_good=num_good,
            num_bad=num_bad,
            num_pending=num_pending,
            mask_agreement=mask_agreement,
        )

        rows.append(dict(
            segment=seg,
            frame=frame,
            step=step,
            num_reviewers=len(reviewers),
            num_good=num_good,
            num_bad=num_bad,
            num_pending=num_pending,
            mask_agreement=mask_agreement,
            **scores,
        ))

    consensus_path = export_root / "consensus.csv"
    if rows:
        with consensus_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Consensus written: %s (%d rows)", consensus_path, len(rows))

    return rows
