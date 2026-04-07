import tempfile
import uuid
import webbrowser
from pathlib import Path
from typing import Optional, Sequence, Tuple

from PyQt5 import QtCore, QtWidgets

from neurochecker.graph import Node
from neurochecker.gui.constants import logger
from neurochecker.gui.mesh import _segment_color_for_plot


def build_plotly_html(
    nodes: Sequence[Node],
    edges: Sequence[Tuple[int, int]],
    *,
    highlight_frame: Optional[int],
    title: str,
    mesh_path: Optional[Path] = None,
    segments: Optional[Sequence[Sequence[int]]] = None,
    segment_colors: Optional[Sequence[str]] = None,
    flagged_points: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> str:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return "<html><body><h3>Plotly is not installed.</h3></body></html>"

    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    zs = [n.z for n in nodes]

    edge_x = []
    edge_y = []
    edge_z = []
    for i, j in edges:
        if i >= len(nodes) or j >= len(nodes):
            continue
        a = nodes[i]
        b = nodes[j]
        edge_x.extend([a.x, b.x, None])
        edge_y.extend([a.y, b.y, None])
        edge_z.extend([a.z, b.z, None])

    fig = go.Figure()
    mesh_color = "rgba(148,94,255,0.85)"
    edge_color = "rgba(0,105,160,0.55)"
    normal_color = "rgba(45,45,45,0.8)"
    branch_color = "rgba(220,60,60,0.95)"
    end_color = "rgba(40,120,255,0.95)"
    highlight_color = "rgba(255,140,0,0.95)"
    flagged_color = "rgba(255,210,40,0.95)"
    if mesh_path and mesh_path.exists():
        logger.info("Loading mesh for plotly: %s", mesh_path)
        verts, faces = None, None
        try:
            import trimesh
            tm = trimesh.load(mesh_path, process=False)
            if isinstance(tm, trimesh.Scene):
                if tm.geometry:
                    tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
                else:
                    tm = None
            if tm is not None and hasattr(tm, "vertices") and hasattr(tm, "faces"):
                verts = tm.vertices
                faces = tm.faces
        except Exception:
            logger.exception("Failed to load mesh for plotly: %s", mesh_path)
        if verts is not None and faces is not None:
            logger.info("Mesh loaded: %s (verts=%d faces=%d)", mesh_path, len(verts), len(faces))
            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=mesh_color,
                    opacity=0.25,
                    name="mesh",
                    showscale=False,
                    flatshading=True,
                    lighting=dict(ambient=0.35, diffuse=0.8, specular=0.2, roughness=0.6, fresnel=0.1),
                    lightposition=dict(x=100, y=200, z=500),
                )
            )
        else:
            logger.warning("Mesh load returned None for %s", mesh_path)
    elif mesh_path:
        logger.warning("Mesh path not found: %s", mesh_path)
    if segments:
        for seg_id, path in enumerate(segments):
            if len(path) < 2:
                continue
            if segment_colors and seg_id < len(segment_colors):
                seg_color = segment_colors[seg_id]
            else:
                seg_color = _segment_color_for_plot(seg_id)
            seg_x = [nodes[idx].x for idx in path if idx < len(nodes)]
            seg_y = [nodes[idx].y for idx in path if idx < len(nodes)]
            seg_z = [nodes[idx].z for idx in path if idx < len(nodes)]
            if len(seg_x) < 2:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=seg_x,
                    y=seg_y,
                    z=seg_z,
                    mode="lines",
                    line=dict(color=seg_color, width=7),
                    name=f"segment {seg_id}",
                )
            )
    elif edge_x:
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=edge_color, width=2),
                name="edges",
            )
        )

    def _node_trace(node_list: Sequence[Node], *, size: int, color: str, name: str) -> Optional["go.Scatter3d"]:
        if not node_list:
            return None
        customdata = [[n.x_px, n.y_px, n.z_frame] for n in node_list]
        return go.Scatter3d(
            x=[n.x for n in node_list],
            y=[n.y for n in node_list],
            z=[n.z for n in node_list],
            mode="markers",
            marker=dict(size=size, color=color),
            name=name,
            customdata=customdata,
            hovertemplate=(
                "phys x=%{x:.3f}<br>"
                "phys y=%{y:.3f}<br>"
                "phys z=%{z:.3f}<br>"
                "px x=%{customdata[0]:.1f}<br>"
                "px y=%{customdata[1]:.1f}<br>"
                "frame=%{customdata[2]}<extra></extra>"
            ),
        )

    normal_nodes = [n for n in nodes if n.label == "normal"]
    branch_nodes = [n for n in nodes if n.label == "branch"]
    end_nodes = [n for n in nodes if n.label == "endpoint"]

    normal_trace = _node_trace(normal_nodes, size=4, color=normal_color, name="skeleton")
    if normal_trace is not None and not segments:
        fig.add_trace(normal_trace)
    branch_trace = _node_trace(branch_nodes, size=6, color=branch_color, name="branch points")
    if branch_trace is not None:
        fig.add_trace(branch_trace)
    end_trace = _node_trace(end_nodes, size=5, color=end_color, name="endpoints")
    if end_trace is not None:
        fig.add_trace(end_trace)

    if flagged_points:
        fig.add_trace(
            go.Scatter3d(
                x=[p[0] for p in flagged_points],
                y=[p[1] for p in flagged_points],
                z=[p[2] for p in flagged_points],
                mode="markers",
                marker=dict(size=8, color=flagged_color),
                name="flagged",
            )
        )

    if highlight_frame is not None:
        highlight_nodes = [n for n in nodes if n.frame == highlight_frame]
        if highlight_nodes:
            fig.add_trace(
                go.Scatter3d(
                    x=[n.x for n in highlight_nodes],
                    y=[n.y for n in highlight_nodes],
                    z=[n.z for n in highlight_nodes],
                    mode="markers",
                    marker=dict(size=7, color=highlight_color),
                    name="current",
                    customdata=[[n.x_px, n.y_px, n.z_frame] for n in highlight_nodes],
                    hovertemplate=(
                        "phys x=%{x:.3f}<br>"
                        "phys y=%{y:.3f}<br>"
                        "phys z=%{z:.3f}<br>"
                        "px x=%{customdata[0]:.1f}<br>"
                        "px y=%{customdata[1]:.1f}<br>"
                        "frame=%{customdata[2]}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=title,
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(248,248,248)",
            aspectmode="data",
        ),
        paper_bgcolor="rgb(248,248,248)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return pio.to_html(fig, include_plotlyjs="inline", full_html=True)


class GraphMapWindow(QtWidgets.QMainWindow):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NeuroChecker Map")
        self.resize(1000, 800)
        self._web_view = None
        self._use_webengine = False
        self._last_html_path: Optional[Path] = None
        self._init_view()

    def _init_view(self) -> None:
        try:
            from PyQt5 import QtWebEngineWidgets  # type: ignore
        except Exception:
            label = QtWidgets.QLabel("PyQtWebEngine not installed. Opening in browser instead.")
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.setCentralWidget(label)
            self._use_webengine = False
            return
        self._web_view = QtWebEngineWidgets.QWebEngineView()
        self.setCentralWidget(self._web_view)
        self._use_webengine = True

    def update_graph(
        self,
        nodes: Sequence[Node],
        edges: Sequence[Tuple[int, int]],
        *,
        highlight_frame: Optional[int] = None,
        title: str = "NeuroChecker Graph",
    ) -> None:
        if not nodes:
            return
        html = build_plotly_html(nodes, edges, highlight_frame=highlight_frame, title=title)
        if self._use_webengine and self._web_view is not None:
            self._web_view.setHtml(html)
            return
        if self._last_html_path is None:
            name = f"neurochecker_map_{uuid.uuid4().hex}.html"
            self._last_html_path = Path(tempfile.gettempdir()) / name
        self._last_html_path.write_text(html, encoding="utf-8")
        webbrowser.open(self._last_html_path.as_uri())
