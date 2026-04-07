from pathlib import Path
from unittest import TestCase

from neurochecker.graph import GraphResult, Node
from neurochecker.gui._mixin_focus import FocusMixin
from neurochecker.hillock import (
    build_soma_aware_segments,
    clone_graph,
    prune_graph_from_hillock,
)


def _node(node_id: int, frame: int) -> Node:
    return Node(
        id=node_id,
        frame=frame,
        run_id="mesh",
        x_px=float(node_id),
        y_px=float(frame),
        z_frame=frame,
        x=float(node_id),
        y=float(frame),
        z=float(frame),
        mask_path="",
        mask_x=0,
        mask_y=0,
        mask_width=0,
        mask_height=0,
        color=None,
    )


class HillockPruneTests(TestCase):
    def test_build_soma_aware_segments_creates_single_soma_branch(self) -> None:
        graph = GraphResult(
            nodes=[_node(0, 0), _node(1, 1), _node(2, 2), _node(3, 3), _node(4, 1), _node(5, 3)],
            edges=[(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)],
            order=[],
            paths=[[0, 1], [1, 2], [1, 4], [2, 3], [2, 5]],
            counts={},
        )

        result = build_soma_aware_segments(graph, hillock_node_id=1, distal_node_id=3)

        self.assertEqual(result.soma_original_node_ids, [0, 1, 4])
        self.assertEqual(result.soma_segment_index, 0)
        self.assertEqual(result.forward_neighbor_original_node_id, 2)
        self.assertEqual(result.segment_nodes[0], [0, 1, 4])
        self.assertEqual(result.segment_edges[0], {(0, 1), (1, 4)})
        self.assertIn(1, result.segment_nodes[result.primary_neurite_segment_index])
        self.assertIn(2, result.segment_nodes[result.primary_neurite_segment_index])
        self.assertEqual(result.current_to_original, [0, 1, 2, 3, 4, 5])
        self.assertEqual(result.original_to_current[5], 5)

    def test_soma_branch_paths_keep_primary_neurite_as_root(self) -> None:
        graph = GraphResult(
            nodes=[_node(0, 0), _node(1, 1), _node(2, 2), _node(3, 3), _node(4, 1), _node(5, 3)],
            edges=[(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)],
            order=[],
            paths=[[0, 1], [1, 2], [1, 4], [2, 3], [2, 5]],
            counts={},
        )

        result = build_soma_aware_segments(graph, hillock_node_id=1, distal_node_id=3)

        class _DummyFocus(FocusMixin):
            pass

        dummy = _DummyFocus()
        dummy.mesh_segments = result.segment_nodes
        dummy.mesh_node_to_segments = {}
        for seg_id, path in enumerate(dummy.mesh_segments):
            for node_id in path:
                dummy.mesh_node_to_segments.setdefault(node_id, []).append(seg_id)
        dummy._primary_neurite_segment_id = result.primary_neurite_segment_index
        dummy._segment_special_paths = {
            result.soma_segment_index: Path("soma"),
            result.primary_neurite_segment_index: Path("root"),
        }
        dummy.active_segment_id = result.soma_segment_index

        paths = {seg_id: path.as_posix() for seg_id, path in dummy._build_segment_tree_paths().items()}

        self.assertEqual(
            paths,
            {
                result.soma_segment_index: "soma",
                result.primary_neurite_segment_index: "root",
                2: "root/branch_1",
                3: "root/branch_2",
            },
        )

    def test_prune_graph_keeps_only_neurite_side_of_hillock(self) -> None:
        graph = GraphResult(
            nodes=[_node(0, 0), _node(1, 1), _node(2, 2), _node(3, 3), _node(4, 1), _node(5, 3)],
            edges=[(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)],
            order=[],
            paths=[],
            counts={},
        )

        result = prune_graph_from_hillock(graph, hillock_node_id=1, distal_node_id=3)

        self.assertEqual(result.kept_original_node_ids, [1, 2, 3, 5])
        self.assertEqual(result.excluded_original_node_ids, [0, 4])
        self.assertEqual(result.forward_neighbor_original_node_id, 2)
        self.assertEqual(result.hillock_current_node_id, 0)
        self.assertEqual(result.distal_current_node_id, 2)
        self.assertEqual(result.forward_neighbor_current_node_id, 1)
        self.assertEqual(len(result.graph.nodes), 4)
        self.assertEqual(result.graph.edges, [(0, 1), (1, 2), (1, 3)])

    def test_prune_graph_requires_distinct_connected_nodes(self) -> None:
        graph = GraphResult(
            nodes=[_node(0, 0), _node(1, 1), _node(2, 2)],
            edges=[(0, 1)],
            order=[],
            paths=[],
            counts={},
        )

        with self.assertRaisesRegex(ValueError, "different"):
            prune_graph_from_hillock(graph, hillock_node_id=1, distal_node_id=1)

        with self.assertRaisesRegex(ValueError, "Could not find"):
            prune_graph_from_hillock(graph, hillock_node_id=0, distal_node_id=2)

    def test_clone_graph_normalizes_node_ids(self) -> None:
        graph = GraphResult(
            nodes=[_node(8, 0), _node(9, 1)],
            edges=[(0, 1)],
            order=[1, 0],
            paths=[[0, 1]],
            counts={"nodes": 2},
        )

        cloned = clone_graph(graph)

        self.assertEqual([node.id for node in cloned.nodes], [0, 1])
        self.assertEqual(cloned.edges, [(0, 1)])
        self.assertEqual(cloned.order, [1, 0])
        self.assertEqual(cloned.paths, [[0, 1]])
        self.assertEqual(cloned.counts, {"nodes": 2})
