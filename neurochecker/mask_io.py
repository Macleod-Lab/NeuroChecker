import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class MaskEntry:
    mask_id: str
    frame: int
    x: int
    y: int
    run_id: str
    path: Path
    width: int
    height: int
    full_width: int
    full_height: int
    color: Optional[Tuple[int, int, int]] = None


@dataclass
class RunStats:
    run_id: str
    max_width: int = 0
    max_height: int = 0
    color: Optional[Tuple[int, int, int]] = None


def load_mask_index(data_root: Path, neuron_id: str) -> List[dict]:
    index_path = data_root / f"neuron_{neuron_id}" / "masks" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Mask index not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Invalid mask index format: {index_path}")
    return data


def load_mask_entries(data_root: Path, neuron_id: str) -> List[MaskEntry]:
    index = load_mask_index(data_root, neuron_id)
    store_dir = data_root / f"neuron_{neuron_id}" / "masks"
    entries: List[MaskEntry] = []
    for entry in index:
        path = store_dir / str(entry.get("path", ""))
        if not path.exists():
            continue
        mask_id = str(entry.get("id") or "").strip()
        if not mask_id:
            mask_id = path.stem
        run_id = str(entry.get("run_id") or "").strip()
        if not run_id:
            continue
        width = int(entry.get("width") or 0)
        height = int(entry.get("height") or 0)
        color = None
        color_entry = entry.get("color")
        if isinstance(color_entry, (list, tuple)) and len(color_entry) >= 3:
            try:
                color = (int(color_entry[0]), int(color_entry[1]), int(color_entry[2]))
            except Exception:
                color = None
        entries.append(
            MaskEntry(
                mask_id=mask_id,
                frame=int(entry.get("frame", 0)),
                x=int(entry.get("x", 0)),
                y=int(entry.get("y", 0)),
                run_id=run_id,
                path=path,
                width=width,
                height=height,
                full_width=int(entry.get("full_width") or 0),
                full_height=int(entry.get("full_height") or 0),
                color=color,
            )
        )
    return entries


def load_mask_array(entry: MaskEntry) -> Optional[np.ndarray]:
    try:
        with np.load(entry.path) as data:
            mask = np.asarray(data["mask"]).astype(np.uint8)
    except Exception:
        return None
    if mask.size == 0:
        return None
    return mask


def iter_mask_entries_by_frame(entries: Iterable[MaskEntry]) -> Dict[int, List[MaskEntry]]:
    by_frame: Dict[int, List[MaskEntry]] = {}
    for entry in entries:
        by_frame.setdefault(entry.frame, []).append(entry)
    return by_frame


def collect_run_stats(entries: Iterable[MaskEntry]) -> Dict[str, RunStats]:
    stats: Dict[str, RunStats] = {}
    for entry in entries:
        run = stats.get(entry.run_id)
        if run is None:
            run = RunStats(run_id=entry.run_id, color=entry.color)
            stats[entry.run_id] = run
        run.max_width = max(run.max_width, entry.width)
        run.max_height = max(run.max_height, entry.height)
        if run.color is None and entry.color is not None:
            run.color = entry.color
    return stats
