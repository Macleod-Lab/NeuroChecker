"""NeuroChecker GUI package."""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

_qt_cache_root = Path(tempfile.gettempdir()) / "neurochecker_qtwebengine"
os.environ.setdefault("QTWEBENGINE_CACHE_PATH", str(_qt_cache_root))
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-gpu-compositing")

from PyQt5 import QtCore, QtWidgets

from neurochecker.gui.main_window import NeuroCheckerWindow

__all__ = ["NeuroCheckerWindow", "main"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    parser = argparse.ArgumentParser(description="NeuroChecker GUI")
    parser.add_argument("--data-root", default=None, help="NeuroTracer data root.")
    parser.add_argument("--images-dir", default=None, help="Image stack directory.")
    parser.add_argument("--skeleton-dir", default=None, help="Directory containing mesh skeleton JSON files.")
    parser.add_argument("--mesh-dir", default=None, help="Directory containing PLY mesh files.")
    parser.add_argument(
        "--mask-editor",
        default=None,
        metavar="EXPORT_DIR",
        help="Launch directly into the Mask Editor with the given export folder.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QtWidgets.QApplication(sys.argv)

    if args.mask_editor:
        from neurochecker.gui.mask_editor import MaskEditorWindow

        window = MaskEditorWindow(export_root=Path(args.mask_editor))
    else:
        window = NeuroCheckerWindow(
            data_root=Path(args.data_root) if args.data_root else None,
            images_dir=Path(args.images_dir) if args.images_dir else None,
            skeleton_dir=Path(args.skeleton_dir) if args.skeleton_dir else None,
            mesh_dir=Path(args.mesh_dir) if args.mesh_dir else None,
        )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
