from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from neurochecker.gui._mixin_data import DataMixin


class _DummyCache:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


class _DummyWindow(DataMixin):
    def __init__(self) -> None:
        self.data_root = Path("C:/tmp")
        self._current_neuron_name = "old_neuron"
        self.entries = ["stale-entry"]
        self.entries_by_frame = {1: ["stale-entry"]}
        self.run_stats = {"run": "stats"}
        self.nodes = ["stale-node"]
        self.nodes_by_frame = {1: ["stale-node"]}
        self.graph = object()
        self.mesh_graph = object()
        self.mesh_segments = [[1, 2]]
        self.mesh_segment_colors = ["red"]
        self.mesh_segment_edges = {0: {(1, 2)}}
        self.mesh_node_to_segments = {1: [0]}
        self.mesh_kdtree = object()
        self.mesh_kdtree_segment_ids = [0]
        self.active_segment_id = 0
        self._segment_nodes = [1, 2]
        self._segment_frame_order = [1]
        self._segment_frame_points_px = [(10.0, 20.0)]
        self._segment_frame_points_xyz = [(1.0, 2.0, 3.0)]
        self._segment_frame_node_ids = [1]
        self._segment_anchor_node_id = 1
        self._mesh_nav_enabled = True
        self._mesh_path = Path("old_mesh.ply")
        self._mesh_preview_points = ["preview"]
        self._mesh_preview_key = ("old_mesh.ply", 1.0)
        self._current_focus_center_px = (10.0, 20.0)
        self._current_focus_source = "stale"
        self.frame_order = [1]
        self.current_frame_index = 0
        self._last_focus = (10.0, 20.0, 256, 256)
        self._last_view_context = ("stale",)
        self.flagged_masks = {"mask_a"}
        self.flagged_points = {(0, 1): {"frame": 1}}
        self._segment_entry_map = {(0, 1): [0]}
        self.mask_cache = _DummyCache()
        self.component_cache = _DummyCache()
        self._flagged_points_path = Path("old_flags.json")
        self.populate_run_list_calls = 0
        self.refresh_navigation_calls = []
        self.update_info_calls = 0
        self.refresh_minimap_calls = 0
        self.rebuild_called = False
        self.load_flagged_points_called = False
        self.resolve_flagged_points_called = False

    def _populate_run_list(self) -> None:
        self.populate_run_list_calls += 1

    def _flagged_points_file_path(self) -> Path:
        return Path(f"{self._current_neuron_name}_viewer_flags.json")

    def _refresh_navigation(self, reset: bool = False) -> None:
        self.refresh_navigation_calls.append(reset)

    def _update_neuron_info_label(self) -> None:
        self.update_info_calls += 1

    def _refresh_minimap(self) -> None:
        self.refresh_minimap_calls += 1

    def _rebuild_graph(self) -> None:
        self.rebuild_called = True

    def _load_flagged_points(self) -> None:
        self.load_flagged_points_called = True

    def _resolve_flagged_points(self) -> None:
        self.resolve_flagged_points_called = True


class LoadNeuronFailureTests(TestCase):
    def test_load_neuron_clears_stale_state_before_raising(self) -> None:
        window = _DummyWindow()

        with patch(
            "neurochecker.gui._mixin_data.load_mask_entries",
            side_effect=FileNotFoundError("missing masks"),
        ), patch(
            "neurochecker.gui._mixin_data.QtWidgets.QMessageBox.warning",
            return_value=None,
        ):
            with self.assertRaises(FileNotFoundError):
                DataMixin._load_neuron(window, "neuron_Broken", raise_on_error=True)

        self.assertEqual(window._current_neuron_name, "neuron_Broken")
        self.assertEqual(window.entries, [])
        self.assertEqual(window.entries_by_frame, {})
        self.assertEqual(window.run_stats, {})
        self.assertEqual(window.nodes, [])
        self.assertEqual(window.nodes_by_frame, {})
        self.assertIsNone(window.graph)
        self.assertIsNone(window.mesh_graph)
        self.assertEqual(window.mesh_segments, [])
        self.assertEqual(window.mesh_segment_colors, [])
        self.assertEqual(window.mesh_segment_edges, {})
        self.assertEqual(window.mesh_node_to_segments, {})
        self.assertIsNone(window.mesh_kdtree)
        self.assertIsNone(window.mesh_kdtree_segment_ids)
        self.assertIsNone(window.active_segment_id)
        self.assertEqual(window._segment_nodes, [])
        self.assertEqual(window._segment_frame_order, [])
        self.assertEqual(window._segment_frame_points_px, [])
        self.assertEqual(window._segment_frame_points_xyz, [])
        self.assertEqual(window._segment_frame_node_ids, [])
        self.assertIsNone(window._segment_anchor_node_id)
        self.assertFalse(window._mesh_nav_enabled)
        self.assertIsNone(window._mesh_path)
        self.assertIsNone(window._mesh_preview_points)
        self.assertIsNone(window._mesh_preview_key)
        self.assertIsNone(window._current_focus_center_px)
        self.assertIsNone(window._current_focus_source)
        self.assertEqual(window.frame_order, [])
        self.assertEqual(window.current_frame_index, 0)
        self.assertIsNone(window._last_focus)
        self.assertIsNone(window._last_view_context)
        self.assertEqual(window.flagged_masks, set())
        self.assertEqual(window.flagged_points, {})
        self.assertEqual(window._segment_entry_map, {})
        self.assertTrue(window.mask_cache.cleared)
        self.assertTrue(window.component_cache.cleared)
        self.assertEqual(window._flagged_points_path, Path("neuron_Broken_viewer_flags.json"))
        self.assertEqual(window.populate_run_list_calls, 1)
        self.assertEqual(window.refresh_navigation_calls, [True])
        self.assertEqual(window.update_info_calls, 1)
        self.assertEqual(window.refresh_minimap_calls, 1)
        self.assertFalse(window.rebuild_called)
        self.assertFalse(window.load_flagged_points_called)
        self.assertFalse(window.resolve_flagged_points_called)
