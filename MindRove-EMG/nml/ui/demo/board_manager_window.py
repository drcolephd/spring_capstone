from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt
from nml.gui_window import GuiWindow
from nml.board_modes import BoardMode
import os
import xml.etree.ElementTree as ET


class BoardManagerWindow(GuiWindow):
    # # # Signals emitted (for BoardManager) # # #
    enable_signal = pyqtSignal(bool)
    assertion_debounce_signal = pyqtSignal(int)
    deassertion_debounce_signal = pyqtSignal(int)
    assertion_threshold_signal = pyqtSignal(float)
    deassertion_threshold_signal = pyqtSignal(float)
    update_mode_signal = pyqtSignal(int)  # emits internal index (mode - 3)
    home_robot = pyqtSignal()

    # Map from mode value (3–8) to label text
    _MODE_LABELS = {
        3: "DPad + A (Press/Latch)",
        4: "DPad + A (Press/Continuous)",
        5: "Directed A-Only (Latch)",
        6: "Non-Directed A-Only (Latch-Toggle)",
        7: "DPad-WINNER + A (Press/Continuous)",
        8: "Wrist LSL (Continuous)",
    }

    # # # UI Elements for setting Parameters # # #
    layout: QtWidgets.QGridLayout | None = None
    asserted_checkbox: QtWidgets.QCheckBox | None = None
    enable_checkbox: QtWidgets.QCheckBox | None = None
    debounce_label: QtWidgets.QLabel | None = None
    assertion_debounce_input_label: QtWidgets.QLabel | None = None
    assertion_debounce_input: QtWidgets.QSpinBox | None = None
    deassertion_debounce_input_label: QtWidgets.QLabel | None = None
    deassertion_debounce_input: QtWidgets.QSpinBox | None = None
    thresholds_label: QtWidgets.QLabel | None = None
    assertion_threshold_input_label: QtWidgets.QLabel | None = None
    assertion_threshold_input: QtWidgets.QDoubleSpinBox | None = None
    deassertion_threshold_input_label: QtWidgets.QLabel | None = None
    deassertion_threshold_input: QtWidgets.QDoubleSpinBox | None = None
    power_threshold_input_label: QtWidgets.QLabel | None = None
    power_threshold_input: QtWidgets.QSpinBox | None = None
    orientation_threshold_input_label: QtWidgets.QLabel | None = None
    orientation_threshold_input: QtWidgets.QDoubleSpinBox | None = None
    update_mode_combo_label: QtWidgets.QLabel | None = None
    update_mode_combo: QtWidgets.QComboBox | None = None
    config_label: QtWidgets.QLabel | None = None
    home_button: QtWidgets.QPushButton | None = None
    save_button: QtWidgets.QPushButton | None = None
    load_button: QtWidgets.QPushButton | None = None
    filename_input: QtWidgets.QLineEdit | None = None

    # Track what modes are allowed for this instance (list of int mode values)
    _allowed_modes: list[int] | None = None

    def __init__(self, board_manager, allowed_modes: list[int] | None = None):
        super().__init__(set_layout=False)
        self._allowed_modes = allowed_modes

        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.setWindowTitle("MindRove Parameters")
        self.setGeometry(500, 250, 450, 700)
        self._initialize_cmu_style()
        self._initialize_grid_widgets(board_manager)
        self.set_modal()
        self.move_to_top_left()

        board_manager.state_change.connect(self.update_state)
        self.show()

    # ------------------------------------------------------------------
    # Public API: allowed modes + selection
    # ------------------------------------------------------------------
    def set_allowed_modes(self, allowed_modes: list[int]) -> None:
        """
        Restrict the combo box to a subset of mode values (3–8).
        The order of allowed_modes determines the combo order.
        """
        self._allowed_modes = allowed_modes
        if self.update_mode_combo is not None:
            self._populate_mode_combo(self._allowed_modes)

    def select_mode(self, mode_value: int) -> None:
        """
        Programmatically select a mode value (3–8) in the combo box,
        if present.
        """
        if self.update_mode_combo is None:
            return
        for idx in range(self.update_mode_combo.count()):
            if self.update_mode_combo.itemData(idx) == mode_value:
                self.update_mode_combo.setCurrentIndex(idx)
                return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _populate_mode_combo(self, allowed_modes: list[BoardMode] | None) -> None:
        """
        Populate the combo box with either all modes or a filtered subset.
        Each itemData holds the actual mode value (3–8).
        """
        self.update_mode_combo.clear()
        if allowed_modes is None:
            modes = list(BoardMode)
        else:
            modes = allowed_modes

        for mode in modes:
            self.update_mode_combo.addItem(mode.label, mode)

    @pyqtSlot(int)
    def _on_mode_index_changed(self, combo_index: int):
        """
        Bridge from combo index to BoardManager.update_mode.
        We emit the internal 'index' expected by BoardManager.set_update_mode:
            internal_index = mode_value - 3
        """
        mode_enum: BoardMode = self.update_mode_combo.itemData(combo_index)
        # BoardManager.set_update_mode still wants 0-based index (mode - 3)
        internal_index = mode_enum.value - 3
        self.update_mode_signal.emit(internal_index)

    # ------------------------------------------------------------------
    # Widget layout
    # ------------------------------------------------------------------
    def _initialize_grid_widgets(self, board_manager):
        # Assertion state checkbox (non-editable)
        self.asserted_checkbox = QtWidgets.QCheckBox("Asserted")
        self.asserted_checkbox.setChecked(board_manager.asserted)  # pyright: ignore[reportOptionalMemberAccess]
        self.asserted_checkbox.setEnabled(False)  # Read-only
        self.layout.addWidget(self.asserted_checkbox, 0, 0, 1, 2)

        # Enable checkbox (editable)
        self.enable_checkbox = QtWidgets.QCheckBox("Enable")
        self.enable_checkbox.setChecked(True)
        self.enable_checkbox.setEnabled(True)
        self.enable_checkbox.clicked.connect(self.set_enabled_state)
        self.layout.addWidget(self.enable_checkbox, 0, 2, 1, 2)

        # Add a header for Debounces
        self.debounce_label = QtWidgets.QLabel("Debounces")
        self.debounce_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.debounce_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.debounce_label, 1, 0, 1, 4)

        # Assertion debounce (in milliseconds)
        self.assertion_debounce_input = QtWidgets.QSpinBox()
        self.assertion_debounce_input.setRange(0, 5000)  # Range 0 to 5000 ms
        self.assertion_debounce_input.valueChanged.connect(board_manager.set_assertion_debounce)
        self.assertion_debounce_input_label = QtWidgets.QLabel("Assertion (ms)")
        self.layout.addWidget(self.assertion_debounce_input_label, 2, 0, 1, 1)
        self.layout.addWidget(self.assertion_debounce_input, 2, 1, 1, 3)

        # Deassertion debounce (in milliseconds)
        self.deassertion_debounce_input = QtWidgets.QSpinBox()
        self.deassertion_debounce_input.setRange(0, 5000)  # Range 0 to 5000 ms
        self.deassertion_debounce_input.valueChanged.connect(board_manager.set_deassertion_debounce)
        self.deassertion_debounce_input_label = QtWidgets.QLabel("Deassertion (ms)")
        self.layout.addWidget(self.deassertion_debounce_input_label, 3, 0, 1, 1)
        self.layout.addWidget(self.deassertion_debounce_input, 3, 1, 1, 3)

        # Add a header for Thresholds
        self.thresholds_label = QtWidgets.QLabel("Thresholds")
        self.thresholds_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.thresholds_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.thresholds_label, 4, 0, 1, 4)

        # Assertion threshold
        self.assertion_threshold_input = QtWidgets.QDoubleSpinBox()
        self.assertion_threshold_input.setRange(0.0, 0.01)
        self.assertion_threshold_input.setSingleStep(0.00001)
        self.assertion_threshold_input.valueChanged.connect(board_manager.set_assertion_threshold)
        self.assertion_threshold_input.setDecimals(5)
        self.assertion_threshold_input_label = QtWidgets.QLabel("Assertion")
        self.layout.addWidget(self.assertion_threshold_input_label, 5, 0, 1, 1)
        self.layout.addWidget(self.assertion_threshold_input, 5, 1, 1, 3)

        # Deassertion threshold
        self.deassertion_threshold_input = QtWidgets.QDoubleSpinBox()
        self.deassertion_threshold_input.setRange(0.0, 1.0)
        self.deassertion_threshold_input.setSingleStep(0.00001)
        self.deassertion_threshold_input.valueChanged.connect(board_manager.set_deassertion_threshold)
        self.deassertion_threshold_input.setDecimals(5)
        self.deassertion_threshold_input_label = QtWidgets.QLabel("Deassertion")
        self.layout.addWidget(self.deassertion_threshold_input_label, 6, 0, 1, 1)
        self.layout.addWidget(self.deassertion_threshold_input, 6, 1, 1, 3)

        # Power threshold
        self.power_threshold_input = QtWidgets.QSpinBox()
        self.power_threshold_input.setRange(0, 10000000)  # Range 0 to 10000000
        self.power_threshold_input.valueChanged.connect(board_manager.set_power_threshold)
        self.power_threshold_input_label = QtWidgets.QLabel("Power")
        self.layout.addWidget(self.power_threshold_input_label, 7, 0, 1, 1)
        self.layout.addWidget(self.power_threshold_input, 7, 1, 1, 3)

        # Orientation threshold
        self.orientation_threshold_input = QtWidgets.QDoubleSpinBox()
        self.orientation_threshold_input.setRange(0, 1.0)
        self.orientation_threshold_input.setSingleStep(0.0005)
        self.orientation_threshold_input.valueChanged.connect(board_manager.set_orientation_threshold)
        self.orientation_threshold_input_label = QtWidgets.QLabel("Orientation")
        self.layout.addWidget(self.orientation_threshold_input_label, 8, 0, 1, 1)
        self.layout.addWidget(self.orientation_threshold_input, 8, 1, 1, 3)

        # Update mode combo (data-driven)
        self.update_mode_combo = QtWidgets.QComboBox()
        self._populate_mode_combo(self._allowed_modes)
        self.update_mode_combo.currentIndexChanged.connect(self._on_mode_index_changed)
        self.update_mode_combo.currentIndexChanged.connect(self.update_valid_parameters)
        self.update_mode_combo_label = QtWidgets.QLabel("Mode")
        self.layout.addWidget(self.update_mode_combo_label, 9, 0, 1, 1)
        self.layout.addWidget(self.update_mode_combo, 9, 1, 1, 3)

        # Add a header for Config interactions
        self.config_label = QtWidgets.QLabel("Configuration File")
        self.config_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.config_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.config_label, 10, 0, 1, 4)

        self.home_button = QtWidgets.QPushButton("Home")
        self.save_button = QtWidgets.QPushButton("Save")
        self.load_button = QtWidgets.QPushButton("Load")
        self.filename_input = QtWidgets.QLineEdit("board_manager_config.xml")

        self.save_button.clicked.connect(lambda: self.save(name=self.filename_input.text()))
        self.load_button.clicked.connect(lambda: self.load(name=self.filename_input.text()))
        self.home_button.clicked.connect(self.send_home_sequence)
        self.layout.addWidget(self.home_button, 11, 0, 1, 1)
        self.layout.addWidget(self.save_button, 11, 1, 1, 1)
        self.layout.addWidget(self.load_button, 11, 2, 1, 1)
        self.layout.addWidget(self.filename_input, 11, 3, 1, 1)

        # Load config (if exists). This will also set the combo index.
        self.load(name="board_manager_config.xml")

        # Ensure parameter visibility matches current combo selection
        self.update_valid_parameters(self.update_mode_combo.currentIndex())

    # ------------------------------------------------------------------
    # Signals -> BoardManager
    # ------------------------------------------------------------------
    @pyqtSlot()
    def send_home_sequence(self):
        self.home_robot.emit()  # pyright: ignore[reportAttributeAccessIssue]

    @pyqtSlot(bool)
    def set_enabled_state(self, en: bool):
        self.enable_signal.emit(en)  # pyright: ignore[reportAttributeAccessIssue]

    @pyqtSlot(bool)
    def update_state(self, state: bool):
        self.asserted_checkbox.setChecked(state)  # pyright: ignore[reportOptionalMemberAccess]

    # ------------------------------------------------------------------
    # Mode-dependent parameter visibility
    # ------------------------------------------------------------------
    @pyqtSlot(int)
    def update_valid_parameters(self, new_index: int):
        """
        Adjust which threshold widgets are visible based on the *mode value*,
        not just index. Mode value is stored in itemData (3–8).
        """
        mode_val = self.update_mode_combo.currentData()
        if mode_val is None:
            mode_val = new_index + 3

        if mode_val == 3:
            self.assertion_threshold_input.setVisible(False)
            self.assertion_threshold_input_label.setVisible(False)
            self.deassertion_threshold_input.setVisible(True)
            self.deassertion_threshold_input_label.setVisible(True)
            self.power_threshold_input.setVisible(True)
            self.power_threshold_input_label.setVisible(True)
        elif mode_val == 4:
            self.assertion_threshold_input.setVisible(False)
            self.assertion_threshold_input_label.setVisible(False)
            self.deassertion_threshold_input.setVisible(True)
            self.deassertion_threshold_input_label.setVisible(True)
            self.power_threshold_input.setVisible(True)
            self.power_threshold_input_label.setVisible(True)
        elif mode_val == 5:
            self.assertion_threshold_input.setVisible(True)
            self.assertion_threshold_input_label.setVisible(True)
            self.deassertion_threshold_input.setVisible(True)
            self.deassertion_threshold_input_label.setVisible(True)
            self.power_threshold_input.setVisible(False)
            self.power_threshold_input_label.setVisible(False)
        elif mode_val == 6:
            self.assertion_threshold_input.setVisible(True)
            self.assertion_threshold_input_label.setVisible(True)
            self.deassertion_threshold_input.setVisible(False)
            self.deassertion_threshold_input_label.setVisible(False)
            self.power_threshold_input.setVisible(False)
            self.power_threshold_input_label.setVisible(False)
        elif mode_val == 7:
            self.assertion_threshold_input.setVisible(False)
            self.assertion_threshold_input_label.setVisible(False)
            self.deassertion_threshold_input.setVisible(True)
            self.deassertion_threshold_input_label.setVisible(True)
            self.power_threshold_input.setVisible(True)
            self.power_threshold_input_label.setVisible(True)
        elif mode_val == 8:
            self.assertion_threshold_input.setVisible(True)
            self.assertion_threshold_input_label.setVisible(True)
            self.deassertion_threshold_input.setVisible(False)
            self.deassertion_threshold_input_label.setVisible(False)
            self.power_threshold_input.setVisible(True)
            self.power_threshold_input_label.setVisible(True)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, name: str = "board_manager_config.xml"):
        """Save the current parameter values to an XML file."""
        os.makedirs(self._configs, exist_ok=True)  # Create the directory if it doesn't exist
        file_path = os.path.join(self._configs, name)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        # Create the XML structure
        board_manager_config = ET.Element("board_manager_config")

        ET.SubElement(board_manager_config, "assertion_debounce").text = str(self.assertion_debounce_input.value())
        ET.SubElement(board_manager_config, "deassertion_debounce").text = str(self.deassertion_debounce_input.value())
        ET.SubElement(board_manager_config, "assertion_threshold").text = str(self.assertion_threshold_input.value())
        ET.SubElement(board_manager_config, "deassertion_threshold").text = str(self.deassertion_threshold_input.value())
        ET.SubElement(board_manager_config, "power_threshold").text = str(self.power_threshold_input.value())
        ET.SubElement(board_manager_config, "orientation_threshold").text = str(self.orientation_threshold_input.value())

        # Store the actual mode value (3–8), not the index
        mode_val = self.update_mode_combo.currentData()
        if mode_val is None:
            mode_val = self.update_mode_combo.currentIndex() + 3
        ET.SubElement(board_manager_config, "update_mode").text = str(mode_val)

        # Write to file
        tree = ET.ElementTree(board_manager_config)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
        print(f"[BoardManagerWindow]::Configuration saved to {file_path}")

    def load(self, name: str = "board_manager_config.xml"):
        """Load parameter values from an XML file."""
        config_file = os.path.join(self._configs, name)  # pyright: ignore[reportAttributeAccessIssue]
        if not os.path.exists(config_file):
            print("[BoardManagerWindow]::Configuration file does not exist!")
            return

        # Parse the XML file
        tree = ET.parse(config_file)
        root = tree.getroot()

        # Load values from XML and set them in the widgets
        self.assertion_debounce_input.setValue(int(root.find("assertion_debounce").text))
        self.deassertion_debounce_input.setValue(int(root.find("deassertion_debounce").text))
        self.assertion_threshold_input.setValue(float(root.find("assertion_threshold").text))
        self.deassertion_threshold_input.setValue(float(root.find("deassertion_threshold").text))
        self.power_threshold_input.setValue(int(root.find("power_threshold").text))
        self.orientation_threshold_input.setValue(float(root.find("orientation_threshold").text))

        update_mode = int(root.find("update_mode").text)

        # Find the combo index that has this mode value in its itemData
        target_index = None
        for idx in range(self.update_mode_combo.count()):
            if self.update_mode_combo.itemData(idx) == update_mode:
                target_index = idx
                break

        if target_index is None:
            print(
                f"[BoardManagerWindow]::Stored mode {update_mode} not in allowed modes "
                f"{self._allowed_modes}; using default index 0."
            )
            target_index = 0

        self.update_mode_combo.setCurrentIndex(target_index)
        print(f"[BoardManagerWindow]::Configuration loaded from {config_file}")
