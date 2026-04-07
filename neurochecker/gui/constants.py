import logging
from typing import Dict, Tuple

from neurochecker.graph import GraphResult

logger = logging.getLogger("neurochecker")

MESH_SKELETON_TARGET_MAX_DIM = 256
MESH_SKELETON_MAX_VOXELS = 20_000_000
MESH_SKELETON_MIN_PITCH = 0.25
MESH_PREVIEW_MAX_POINTS = 16000
MESH_PREVIEW_LOCAL_MAX_POINTS = 8000
MASK_OUTLINE_THICKNESS = 2
FLAG_FRAME_BORDER_PX = 4
SEGMENT_RATIO_SCALE = 1.5
SAMPLE_EXPORT_MIN_LONG_SIDE_PX = 1024
_MESH_SKELETON_CACHE: Dict[Tuple[str, float, float, float, float], GraphResult] = {}
