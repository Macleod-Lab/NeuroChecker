import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from scipy.spatial import cKDTree

from neurochecker.graph import GraphResult, Node
from neurochecker.gui.data import (
    ComponentCache,
    ImageSampler,
    MaskCache,
)
from neurochecker.gui.full_skeleton_window import FullSkeleton3DWindow
from neurochecker.gui.plotly_map import GraphMapWindow
from neurochecker.gui.widgets import MiniMap3DWidget, SegmentBarWidget
from neurochecker.mask_io import MaskEntry

from neurochecker.gui._mixin_ui import UiMixin
from neurochecker.gui._mixin_editing import EditingMixin
from neurochecker.gui._mixin_navigation import NavigationMixin
from neurochecker.gui._mixin_rendering import RenderingMixin
from neurochecker.gui._mixin_focus import FocusMixin
from neurochecker.gui._mixin_minimap import MinimapMixin
from neurochecker.gui._mixin_export import ExportMixin
from neurochecker.gui._mixin_data import DataMixin


class NeuroCheckerWindow(
    UiMixin,
    EditingMixin,
    NavigationMixin,
    RenderingMixin,
    FocusMixin,
    MinimapMixin,
    ExportMixin,
    DataMixin,
    QtWidgets.QMainWindow,
):
    def __init__(self, *, data_root: Optional[Path] = None, images_dir: Optional[Path] = None, skeleton_dir: Optional[Path] = None, mesh_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.setWindowTitle("NeuroChecker")
        self.resize(1600, 1000)

        self.data_root: Optional[Path] = data_root
        self.images_dir: Optional[Path] = images_dir
        self.skeleton_dir: Optional[Path] = skeleton_dir
        self.mesh_dir: Optional[Path] = mesh_dir
        self.entries: List[MaskEntry] = []
        self.skeleton_data = {}
        self.use_skeletons = False
        self.skeleton_path: Optional[Path] = None
        self.mesh_path: Optional[Path] = None
        self.entries_by_frame: Dict[int, List[MaskEntry]] = {}
        self.run_stats = {}
        self._base_graph: Optional[GraphResult] = None
        self.nodes: List[Node] = []
        self.nodes_by_frame: Dict[int, List[Node]] = {}
        self.graph: Optional[GraphResult] = None
        self.mesh_graph: Optional[GraphResult] = None
        self._current_original_node_ids: List[int] = []
        self._original_to_current_node_ids: Dict[int, int] = {}
        self.mesh_segments: List[List[int]] = []
        self.mesh_segment_colors: List[str] = []
        self.mesh_segment_edges: Dict[int, Set[Tuple[int, int]]] = {}
        self.mesh_node_to_segments: Dict[int, List[int]] = {}
        self.mesh_kdtree: Optional[cKDTree] = None
        self.mesh_kdtree_segment_ids: Optional[np.ndarray] = None
        self.active_segment_id: Optional[int] = None
        self._segment_nodes: List[int] = []
        self._segment_frame_order: List[int] = []
        self._segment_frame_points_px: List[Tuple[float, float]] = []
        self._segment_frame_points_xyz: List[Tuple[float, float, float]] = []
        self._segment_frame_node_ids: List[Optional[int]] = []
        self._segment_anchor_node_id: Optional[int] = None
        self._mesh_nav_enabled = False
        self._mesh_path: Optional[Path] = None
        self._mesh_preview_points: Optional[np.ndarray] = None
        self._mesh_preview_key: Optional[Tuple[str, float]] = None
        self._current_neuron_name: Optional[str] = None
        self.flagged_points: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._flagged_points_path: Optional[Path] = None
        self._edit_mode = False
        self._edit_brush_add = True
        self._edit_brush_size = 8
        self._edit_active_entry: Optional[MaskEntry] = None
        self._edit_active_mask: Optional[np.ndarray] = None
        self._edit_active_frame: Optional[int] = None
        self._edit_dirty = False
        self._edit_index_dirty = False
        self._edit_painting = False
        self._edit_last_pos: Optional[Tuple[int, int]] = None
        self._brush_cache: Dict[int, np.ndarray] = {}
        self._branch_options: List[int] = []
        self._branch_node_id: Optional[int] = None
        self._branch_hint: Optional[str] = None
        self._branch_option_buttons: List[QtWidgets.QPushButton] = []
        self._current_focus_center_px: Optional[Tuple[float, float]] = None
        self._current_focus_source: Optional[str] = None
        self.frame_order: List[int] = []
        self.current_frame_index = 0
        self.flagged_masks: Set[str] = set()
        self.mask_cache = MaskCache()
        self.component_cache = ComponentCache()
        self._segment_entry_map: Dict[Tuple[int, int], List[int]] = {}
        self.image_sampler: Optional[ImageSampler] = None
        self._populating_runs = False
        self._resize_timer = QtCore.QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._refresh_current_view)
        self._minimap_current_items: List[QtWidgets.QGraphicsEllipseItem] = []
        self.map_window: Optional[GraphMapWindow] = None
        self.skeleton_3d_window: Optional[FullSkeleton3DWindow] = None
        self._minimap_zoom = 1.0
        self._minimap_frame_window = 4
        self._minimap_edge_median_px = 0.0
        self._last_focus: Optional[Tuple[float, float, int, int]] = None
        self._minimap_local_k = 40
        self._map_html_path: Optional[Path] = None
        self._overlay_timer = QtCore.QTimer()
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.timeout.connect(self._apply_pending_overlay)
        self._last_view_context: Optional[Tuple[int, Tuple[int, int, int, int], Tuple[int, int], np.ndarray, float]] = None
        self.minimap_widget: Optional[MiniMap3DWidget] = None
        self.segment_bar: Optional[SegmentBarWidget] = None
        self._hillock_original_node_id: Optional[int] = None
        self._distal_original_node_id: Optional[int] = None
        self._hillock_forward_original_node_id: Optional[int] = None
        self._soma_original_node_ids: List[int] = []
        self._soma_segment_id: Optional[int] = None
        self._primary_neurite_segment_id: Optional[int] = None
        self._segment_special_paths: Dict[int, Path] = {}
        self._hillock_cutoff_path: Optional[Path] = None

        self._build_ui()
        self._load_initial_state()
