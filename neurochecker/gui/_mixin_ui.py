from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets

from neurochecker.gui.widgets import SegmentBarWidget


class UiMixin:
    def _build_menu_bar(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        mask_editor_action = file_menu.addAction("Mask Editor...")
        mask_editor_action.triggered.connect(self._open_mask_editor)

    def _open_mask_editor(self) -> None:
        from neurochecker.gui.mask_editor import MaskEditorWindow

        editor = MaskEditorWindow(parent=None)
        editor.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        editor.show()
        self._mask_editor_window = editor

    def _build_ui(self) -> None:
        self._build_menu_bar()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_panel.setFixedWidth(330)

        data_root_group = QtWidgets.QGroupBox("NeuroTracer Data Root")
        data_root_layout = QtWidgets.QHBoxLayout(data_root_group)
        self.data_root_edit = QtWidgets.QLineEdit()
        browse_root_btn = QtWidgets.QPushButton("Browse")
        browse_root_btn.clicked.connect(self._browse_data_root)
        data_root_layout.addWidget(self.data_root_edit)
        data_root_layout.addWidget(browse_root_btn)
        left_layout.addWidget(data_root_group)

        images_group = QtWidgets.QGroupBox("Image Stack Directory")
        images_layout = QtWidgets.QHBoxLayout(images_group)
        self.images_dir_edit = QtWidgets.QLineEdit()
        browse_images_btn = QtWidgets.QPushButton("Browse")
        browse_images_btn.clicked.connect(self._browse_images_dir)
        images_layout.addWidget(self.images_dir_edit)
        images_layout.addWidget(browse_images_btn)
        left_layout.addWidget(images_group)

        skeleton_group = QtWidgets.QGroupBox("Skeletons Directory (Alternative)")
        skeleton_layout = QtWidgets.QHBoxLayout(skeleton_group)
        self.skeleton_dir_edit = QtWidgets.QLineEdit()
        browse_skeleton_btn = QtWidgets.QPushButton("Browse")
        browse_skeleton_btn.clicked.connect(self._browse_skeleton_dir)
        skeleton_layout.addWidget(self.skeleton_dir_edit)
        skeleton_layout.addWidget(browse_skeleton_btn)
        left_layout.addWidget(skeleton_group)

        mesh_group = QtWidgets.QGroupBox("Meshes Directory (Optional)")
        mesh_layout = QtWidgets.QHBoxLayout(mesh_group)
        self.mesh_dir_edit = QtWidgets.QLineEdit()
        browse_mesh_btn = QtWidgets.QPushButton("Browse")
        browse_mesh_btn.clicked.connect(self._browse_mesh_dir)
        mesh_layout.addWidget(self.mesh_dir_edit)
        mesh_layout.addWidget(browse_mesh_btn)
        left_layout.addWidget(mesh_group)

        self.neuron_list = QtWidgets.QListWidget()
        self.neuron_list.itemSelectionChanged.connect(self._on_neuron_selected)
        neuron_group = QtWidgets.QGroupBox("Neurons")
        neuron_layout = QtWidgets.QVBoxLayout(neuron_group)
        neuron_layout.addWidget(self.neuron_list)
        self.neuron_info_label = QtWidgets.QLabel("")
        self.neuron_info_label.setWordWrap(True)
        self.neuron_info_label.setStyleSheet("color: #aaa; font-size: 11px; padding: 2px 0;")
        neuron_layout.addWidget(self.neuron_info_label)
        neuron_props_btn = QtWidgets.QPushButton("Neuron Properties...")
        neuron_props_btn.clicked.connect(self._open_neuron_properties)
        neuron_layout.addWidget(neuron_props_btn)
        left_layout.addWidget(neuron_group)

        self.run_list = QtWidgets.QListWidget()
        self.run_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        run_group = QtWidgets.QGroupBox("Mask Runs")
        run_layout = QtWidgets.QVBoxLayout(run_group)
        run_layout.addWidget(self.run_list)
        self.merge_runs_btn = QtWidgets.QPushButton("Merge Selected Runs")
        self.merge_runs_btn.setToolTip("Combine selected runs into a single run (OR masks per frame)")
        self.merge_runs_btn.clicked.connect(self._merge_selected_runs)
        self.merge_runs_btn.setEnabled(False)
        run_layout.addWidget(self.merge_runs_btn)
        self.run_list.itemSelectionChanged.connect(self._on_run_selection_changed)
        left_layout.addWidget(run_group)

        settings_group = QtWidgets.QGroupBox("Graph Settings")
        settings_layout = QtWidgets.QFormLayout(settings_group)
        self.pixel_xy_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_xy_spin.setDecimals(6)
        self.pixel_xy_spin.setRange(0.000001, 1000.0)
        self.pixel_xy_spin.setValue(0.00495)
        settings_layout.addRow("Pixel size (XY):", self.pixel_xy_spin)

        self.slice_z_spin = QtWidgets.QDoubleSpinBox()
        self.slice_z_spin.setDecimals(6)
        self.slice_z_spin.setRange(0.000001, 1000.0)
        self.slice_z_spin.setValue(0.059)
        settings_layout.addRow("Slice thickness (Z):", self.slice_z_spin)

        self.graph_mode_combo = QtWidgets.QComboBox()
        self.graph_mode_combo.addItems(["mst", "frame_link"])
        settings_layout.addRow("Graph mode:", self.graph_mode_combo)

        self.k_neighbors_spin = QtWidgets.QSpinBox()
        self.k_neighbors_spin.setRange(1, 50)
        self.k_neighbors_spin.setValue(20)
        settings_layout.addRow("k neighbors:", self.k_neighbors_spin)

        self.max_frame_gap_spin = QtWidgets.QSpinBox()
        self.max_frame_gap_spin.setRange(0, 1000)
        self.max_frame_gap_spin.setValue(8)
        settings_layout.addRow("Max frame gap:", self.max_frame_gap_spin)

        self.max_distance_spin = QtWidgets.QDoubleSpinBox()
        self.max_distance_spin.setDecimals(6)
        self.max_distance_spin.setRange(0.0, 1e6)
        self.max_distance_spin.setValue(0.0)
        settings_layout.addRow("Max distance (0=off):", self.max_distance_spin)

        self.focus_mode_combo = QtWidgets.QComboBox()
        self.focus_mode_combo.addItem("Largest mask", userData="largest")
        self.focus_mode_combo.addItem("Nearest to previous", userData="nearest")
        self.focus_mode_combo.addItem("Centroid of frame", userData="centroid")
        self.focus_mode_combo.currentIndexChanged.connect(self._refresh_current_view)
        settings_layout.addRow("Focus mode:", self.focus_mode_combo)

        self.focus_run_combo = QtWidgets.QComboBox()
        self.focus_run_combo.addItem("Auto (all runs)", userData=None)
        self.focus_run_combo.currentIndexChanged.connect(self._on_focus_run_changed)
        settings_layout.addRow("Focus run:", self.focus_run_combo)

        rebuild_btn = QtWidgets.QPushButton("Rebuild Graph")
        rebuild_btn.clicked.connect(self._rebuild_graph)
        settings_layout.addRow(rebuild_btn)
        reconcile_btn = QtWidgets.QPushButton("Reconcile Disconnected...")
        reconcile_btn.setToolTip("Bridge or remove disconnected skeleton components")
        reconcile_btn.clicked.connect(self._open_reconcile_dialog)
        settings_layout.addRow(reconcile_btn)
        left_layout.addWidget(settings_group)

        hillock_group = QtWidgets.QGroupBox("Hillock / Soma Cutoff")
        hillock_layout = QtWidgets.QVBoxLayout(hillock_group)
        hillock_help = QtWidgets.QLabel(
            "Right-click a node in the 3D minimap to set the hillock or a distal neurite node. "
            "Everything behind the hillock is treated as soma and removed from the working skeleton."
        )
        hillock_help.setWordWrap(True)
        hillock_help.setStyleSheet("color: #aaa; font-size: 11px;")
        hillock_layout.addWidget(hillock_help)
        hillock_btn_row = QtWidgets.QHBoxLayout()
        self.hillock_current_btn = QtWidgets.QPushButton("Current → Hillock")
        self.hillock_current_btn.clicked.connect(self._set_hillock_from_current_node)
        self.distal_current_btn = QtWidgets.QPushButton("Current → Distal")
        self.distal_current_btn.clicked.connect(self._set_distal_from_current_node)
        hillock_btn_row.addWidget(self.hillock_current_btn)
        hillock_btn_row.addWidget(self.distal_current_btn)
        hillock_layout.addLayout(hillock_btn_row)
        self.open_hillock_picker_btn = QtWidgets.QPushButton("Open Full 3D Picker")
        self.open_hillock_picker_btn.clicked.connect(self._open_full_skeleton_window)
        hillock_layout.addWidget(self.open_hillock_picker_btn)
        self.clear_hillock_btn = QtWidgets.QPushButton("Clear Cutoff")
        self.clear_hillock_btn.clicked.connect(self._clear_hillock_cutoff)
        hillock_layout.addWidget(self.clear_hillock_btn)
        self.hillock_status_label = QtWidgets.QLabel("Hillock not set | Distal not set | Full skeleton")
        self.hillock_status_label.setWordWrap(True)
        self.hillock_status_label.setStyleSheet("color: #ddd; font-size: 11px; padding-top: 2px;")
        hillock_layout.addWidget(self.hillock_status_label)
        left_layout.addWidget(hillock_group)

        crop_group = QtWidgets.QGroupBox("Overlay")
        crop_layout = QtWidgets.QFormLayout(crop_group)
        self.overlay_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.overlay_alpha.setRange(0, 255)
        self.overlay_alpha.setValue(80)
        self.overlay_alpha.valueChanged.connect(self._refresh_current_view)
        crop_layout.addRow("Mask alpha:", self.overlay_alpha)
        self.fast_scrub_check = QtWidgets.QCheckBox("Fast scrub (defer overlay)")
        self.fast_scrub_check.setChecked(True)
        self.fast_scrub_check.setToolTip("Render the image first, then draw masks after a short pause.")
        self.fast_scrub_check.stateChanged.connect(self._refresh_current_view)
        crop_layout.addRow(self.fast_scrub_check)
        left_layout.addWidget(crop_group)

        edit_group = QtWidgets.QGroupBox("Mask Editing")
        edit_layout = QtWidgets.QFormLayout(edit_group)
        self.edit_mode_check = QtWidgets.QCheckBox("Enable edit (E)")
        self.edit_mode_check.stateChanged.connect(self._toggle_edit_mode)
        edit_layout.addRow(self.edit_mode_check)
        brush_mode_layout = QtWidgets.QHBoxLayout()
        self.edit_brush_add_btn = QtWidgets.QRadioButton("Add")
        self.edit_brush_erase_btn = QtWidgets.QRadioButton("Erase")
        self.edit_brush_add_btn.setChecked(True)
        self.edit_brush_add_btn.toggled.connect(self._set_brush_mode)
        self.edit_brush_erase_btn.toggled.connect(self._set_brush_mode)
        brush_mode_layout.addWidget(self.edit_brush_add_btn)
        brush_mode_layout.addWidget(self.edit_brush_erase_btn)
        edit_layout.addRow("Brush mode:", brush_mode_layout)
        self.edit_brush_size_spin = QtWidgets.QSpinBox()
        self.edit_brush_size_spin.setRange(1, 100)
        self.edit_brush_size_spin.setValue(self._edit_brush_size)
        self.edit_brush_size_spin.valueChanged.connect(self._set_brush_size)
        edit_layout.addRow("Brush size:", self.edit_brush_size_spin)
        self.edit_dust_size_spin = QtWidgets.QSpinBox()
        self.edit_dust_size_spin.setRange(1, 5000)
        self.edit_dust_size_spin.setValue(25)
        edit_layout.addRow("Min component px:", self.edit_dust_size_spin)
        self.remove_dust_btn = QtWidgets.QPushButton("Remove dust")
        self.remove_dust_btn.clicked.connect(self._remove_dust_current_mask)
        edit_layout.addRow(self.remove_dust_btn)
        self.smooth_kernel_spin = QtWidgets.QSpinBox()
        self.smooth_kernel_spin.setRange(1, 51)
        self.smooth_kernel_spin.setSingleStep(2)
        self.smooth_kernel_spin.setValue(3)
        edit_layout.addRow("Smooth kernel:", self.smooth_kernel_spin)
        self.smooth_mask_btn = QtWidgets.QPushButton("Smooth mask")
        self.smooth_mask_btn.clicked.connect(self._smooth_current_mask)
        edit_layout.addRow(self.smooth_mask_btn)
        self.fill_holes_btn = QtWidgets.QPushButton("Fill holes (active mask)")
        self.fill_holes_btn.clicked.connect(self._fill_holes_current_mask)
        edit_layout.addRow(self.fill_holes_btn)
        self.fill_holes_all_btn = QtWidgets.QPushButton("Fill holes (all masks)")
        self.fill_holes_all_btn.clicked.connect(self._fill_holes_all_masks)
        edit_layout.addRow(self.fill_holes_all_btn)
        self.remove_dust_all_btn = QtWidgets.QPushButton("Remove dust (all masks)")
        self.remove_dust_all_btn.clicked.connect(self._remove_dust_all_masks)
        edit_layout.addRow(self.remove_dust_all_btn)
        self.smooth_all_btn = QtWidgets.QPushButton("Smooth mask (all masks)")
        self.smooth_all_btn.clicked.connect(self._smooth_all_masks)
        edit_layout.addRow(self.smooth_all_btn)
        left_layout.addWidget(edit_group)

        qc_group = QtWidgets.QGroupBox("QC / Export")
        qc_layout = QtWidgets.QVBoxLayout(qc_group)
        self.flag_btn = QtWidgets.QPushButton("Flag current frame (F)")
        self.flag_btn.clicked.connect(self._toggle_flag_frame)
        self.export_flags_btn = QtWidgets.QPushButton("Export flags CSV")
        self.export_flags_btn.clicked.connect(self._export_flags_csv)
        self.export_crops_btn = QtWidgets.QPushButton("Export crops + masks")
        self.export_crops_btn.clicked.connect(self._export_crops_and_masks)
        self.mass_export_btn = QtWidgets.QPushButton("Mass Export All Neurons...")
        self.mass_export_btn.clicked.connect(self._mass_export)
        qc_layout.addWidget(self.flag_btn)
        qc_layout.addWidget(self.export_flags_btn)
        qc_layout.addWidget(self.export_crops_btn)
        qc_layout.addWidget(self.mass_export_btn)
        left_layout.addWidget(qc_group)

        left_layout.addStretch()

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        self.scene = QtWidgets.QGraphicsScene()
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.view.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        right_layout.addWidget(self.view, 1)

        self._build_minimap()
        if self.minimap_widget is not None:
            self.minimap_widget.nodeContextRequested.connect(self._show_minimap_node_context_menu)

        nav_layout = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.prev_btn.setToolTip("Previous frame. At branch dead-ends, auto-follows if there is one branch choice.")
        self.prev_btn.clicked.connect(self._prev_frame)
        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.setToolTip("Next frame. At branch dead-ends, auto-follows if there is one branch choice.")
        self.next_btn.clicked.connect(self._next_frame)
        self.node_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.node_slider.setRange(0, 0)
        self.node_slider.valueChanged.connect(self._on_slider_changed)
        self.node_label = QtWidgets.QLabel("Frame: -")
        self.node_label.setWordWrap(True)
        self.node_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.node_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.node_label.setMinimumHeight(36)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.node_slider, 1)
        nav_layout.addWidget(self.node_label)
        right_layout.addLayout(nav_layout)
        self.ratio_label = QtWidgets.QLabel("")
        self.ratio_label.setWordWrap(True)
        self.ratio_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.ratio_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.ratio_label.setMinimumHeight(24)
        right_layout.addWidget(self.ratio_label)
        goto_layout = QtWidgets.QHBoxLayout()
        goto_layout.setContentsMargins(0, 0, 0, 0)
        goto_label = QtWidgets.QLabel("Go to frame:")
        self.goto_frame_spin = QtWidgets.QSpinBox()
        self.goto_frame_spin.setRange(0, 99999)
        self.goto_frame_spin.setKeyboardTracking(False)
        goto_go_btn = QtWidgets.QPushButton("Go")
        goto_go_btn.setFixedWidth(40)
        goto_go_btn.clicked.connect(self._goto_frame)
        self.goto_frame_spin.editingFinished.connect(self._goto_frame)
        goto_layout.addWidget(goto_label)
        goto_layout.addWidget(self.goto_frame_spin)
        goto_layout.addWidget(goto_go_btn)
        goto_layout.addStretch()
        right_layout.addLayout(goto_layout)
        self.branch_nav_widget = QtWidgets.QWidget()
        branch_nav_layout = QtWidgets.QHBoxLayout(self.branch_nav_widget)
        branch_nav_layout.setContentsMargins(0, 0, 0, 0)
        branch_nav_layout.setSpacing(6)
        self.branch_nav_label = QtWidgets.QLabel("Branches: -")
        self.branch_nav_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.branch_nav_label.setMinimumWidth(110)
        branch_nav_layout.addWidget(self.branch_nav_label)
        self.branch_nav_buttons_layout = QtWidgets.QHBoxLayout()
        self.branch_nav_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.branch_nav_buttons_layout.setSpacing(4)
        branch_nav_layout.addLayout(self.branch_nav_buttons_layout, 1)
        right_layout.addWidget(self.branch_nav_widget)
        self.segment_bar = SegmentBarWidget()
        self.segment_bar.setVisible(False)
        self.segment_bar.segmentClicked.connect(self._on_segment_bar_clicked)
        right_layout.addWidget(self.segment_bar)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.status_label.setStyleSheet("padding: 2px 6px;")
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        status.addPermanentWidget(self.status_label, 1)

        layout.addWidget(left_panel)
        layout.addWidget(right_panel, 1)

        self._init_shortcuts()

    def _load_initial_state(self) -> None:
        if self.data_root:
            self.data_root_edit.setText(str(self.data_root))
            self._populate_neuron_list()
        if self.images_dir:
            self.images_dir_edit.setText(str(self.images_dir))
            self._init_image_sampler()
        if self.skeleton_dir:
            self.skeleton_dir_edit.setText(str(self.skeleton_dir))
            self._load_skeletons_from_dir(self.skeleton_dir)
        if self.mesh_dir:
            self.mesh_dir_edit.setText(str(self.mesh_dir))
        QtCore.QTimer.singleShot(0, self._position_minimap)

    def _init_shortcuts(self) -> None:
        self._shortcut_prev = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)
        self._shortcut_prev.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_prev.activated.connect(self._prev_frame)

        self._shortcut_next = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        self._shortcut_next.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_next.activated.connect(self._next_frame)

        self._shortcut_prev_alt = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_A), self)
        self._shortcut_prev_alt.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_prev_alt.activated.connect(self._prev_frame)

        self._shortcut_next_alt = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D), self)
        self._shortcut_next_alt.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_next_alt.activated.connect(self._next_frame)
        self._shortcut_map = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_M), self)
        self._shortcut_map.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_map.activated.connect(self._open_map_window)
        self._shortcut_full_picker = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+M"), self)
        self._shortcut_full_picker.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_full_picker.activated.connect(self._open_full_skeleton_window)
        self._shortcut_flag = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F), self)
        self._shortcut_flag.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_flag.activated.connect(self._toggle_flag_frame)
        self._shortcut_edit = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_E), self)
        self._shortcut_edit.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_edit.activated.connect(self._toggle_edit_shortcut)
        self._shortcut_props = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+I"), self)
        self._shortcut_props.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_props.activated.connect(self._open_neuron_properties)
        self._shortcut_goto = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+G"), self)
        self._shortcut_goto.setContext(QtCore.Qt.ApplicationShortcut)
        self._shortcut_goto.activated.connect(self._focus_goto_frame)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
                if not self._focus_allows_navigation():
                    return False
                self._prev_frame()
                return True
            if key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
                if not self._focus_allows_navigation():
                    return False
                self._next_frame()
                return True
            if QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_9:
                if not self._focus_allows_navigation():
                    return False
                option_index = key - QtCore.Qt.Key_1
                self._select_branch_option(option_index)
                return True
        if obj is self.view.viewport() and self._edit_mode:
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                self._edit_painting = True
                self._edit_last_pos = None
                self._apply_edit_from_event(event)
                return True
            if event.type() == QtCore.QEvent.MouseMove and self._edit_painting:
                if event.buttons() & QtCore.Qt.LeftButton:
                    self._apply_edit_from_event(event)
                    return True
            if event.type() == QtCore.QEvent.MouseButtonRelease and self._edit_painting:
                self._edit_painting = False
                self._edit_last_pos = None
                self._commit_edit_mask()
                return True
        return super().eventFilter(obj, event)

    def _focus_allows_navigation(self) -> bool:
        widget = QtWidgets.QApplication.focusWidget()
        if widget is None:
            return True
        if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QTextEdit)):
            return False
        return True

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self._prev_frame()
            return
        if event.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self._next_frame()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_minimap()
        self._resize_timer.start(150)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._position_minimap()
        self._refresh_minimap()

    def _browse_skeleton_dir(self) -> None:
        """Browse for skeletons directory."""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Skeletons Directory")
        if dir_path:
            self.skeleton_dir = Path(dir_path)
            self.skeleton_dir_edit.setText(dir_path)
            self._load_skeletons_from_dir(self.skeleton_dir)

    def _browse_mesh_dir(self) -> None:
        """Browse for meshes directory."""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Meshes Directory")
        if dir_path:
            self.mesh_dir = Path(dir_path)
            self.mesh_dir_edit.setText(dir_path)

    def _load_skeleton_from_ui(self) -> None:
        """Load skeleton from the UI text field."""
        skeleton_path_str = self.skeleton_edit.text().strip()
        if not skeleton_path_str:
            QtWidgets.QMessageBox.warning(self, "Load Skeleton", "Please enter a skeleton file path.")
            return
        try:
            skeleton_path = Path(skeleton_path_str)
            if not skeleton_path.exists():
                QtWidgets.QMessageBox.warning(self, "Load Skeleton", f"Skeleton file does not exist: {skeleton_path}")
                return
            self.skeleton_path = skeleton_path
            self._load_mesh_skeleton(skeleton_path)
            # Also load mesh if specified
            mesh_path_str = self.mesh_edit.text().strip()
            if mesh_path_str:
                mesh_path = Path(mesh_path_str)
                if mesh_path.exists():
                    self.mesh_path = mesh_path
                    self._load_mesh_for_visualization(mesh_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load Skeleton", f"Failed to load skeleton: {exc}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self.image_sampler is not None:
            self.image_sampler.shutdown()
        super().closeEvent(event)
