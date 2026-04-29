from nml.gui_window import GuiWindow
from nml.model_interactor import ModelInteractor
from nml.feature_weights import model as default_model
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
import os, math, time
from typing import Tuple
from nml.vigem import ViGEmPy, ViGEmEnum
from pynput.keyboard import Controller, Key
from collections import deque

class FittsGridStateMachine(QObject):
    # Signals for task events
    running: bool = False
    target_set = pyqtSignal(int, int)  # Target acquired
    target_hold_completed = pyqtSignal()
    task_completed = pyqtSignal(object, float)
    trial_completed = pyqtSignal(int, float)
    moving = pyqtSignal()
    resting = pyqtSignal()

    gamepad: ViGEmPy = None
    trial_duration: "np.ndarray[np.float32]" = None
    _move_start_ms: float = 0.0
    _total_trials: int = 0
    _current_trial: int = 0
    _x: float = None
    _y: float = None
    _delta: float = 0.0
    _delta_threshold: float = 0.000025 # Threshold for moving vs rest
    _at_rest: bool = True

    _a_button_bytecode = ViGEmEnum.encode(["A"])
    _up_button_bytecode = ViGEmEnum.encode(["D-Pad Up"])
    _down_button_bytecode = ViGEmEnum.encode(["D-Pad Down"])
    _left_button_bytecode = ViGEmEnum.encode(["D-Pad Left"])
    _right_button_bytecode = ViGEmEnum.encode(["D-Pad Right"])
    _assertion_threshold: float = 0.0016
    _deassertion_threshold: float = 0.04
    _power_threshold: float = 5000000

    def __init__(self, grid_size: int = 5, num_trials: int = 20, hold_duration_ms: int = 1000, parent=None):
        super().__init__(parent)
        self.grid_size = grid_size
        self._total_trials = num_trials
        self.entropy = math.log2(2 ** (2 * grid_size))  # Shannon entropy in bits
        self.current_target = None
        self.hold_duration_ms = hold_duration_ms
        self._hold_timer = QTimer(self)  # Timer for hold check
        self._hold_timer.setInterval(self.hold_duration_ms)
        self._hold_timer.timeout.connect(self.on_target_hold_complete)
        self._is_cursor_in_target = False
        self.gamepad = ViGEmPy()
        self.set_total_trials(num_trials)

    # def update_gamepad(self, dx, dy, power):
    #     btn = 0x0000 # Default byte code is all button bytes off.
    #     if (dx*dx) > (dy*dy): 
    #         if dx > self._deassertion_threshold:
    #             btn += self._right_button_bytecode
    #         elif dx < -self._deassertion_threshold:
    #             btn += self._left_button_bytecode
    #     else:
    #         if dy > self._deassertion_threshold:
    #             btn += self._up_button_bytecode
    #         elif dy < -self._deassertion_threshold:
    #             btn += self._down_button_bytecode
    #     # print(power) # For debug
    #     if power > self._power_threshold:
    #         btn += self._a_button_bytecode
    #     self.gamepad.send_input(btn)

    def start_task(self, x: float, y: float):
        """Starts the task by selecting the first target and setting the state."""
        self.select_new_target()
        self._x = x
        self._y = y
        self._move_start_ms = time.time_ns() / 1_000_000
        self._current_trial = 0
        self.trial_duration = np.full(self._total_trials, 30.0).astype(np.float32)
        self.running = True

    def stop_task(self):
        """Stops the task and resets state."""
        self.running = False
        if self._hold_timer.isActive():
            self._hold_timer.stop()
        self.task_completed.emit(self.trial_duration, self.entropy)

    def set_total_trials(self, num_trials: int):
        self._total_trials = num_trials
        self._current_trial = 0
        self.trial_duration = np.full(num_trials, 30.0).astype(np.float32)

    def select_new_target(self):
        """Randomly selects a new target and emits the target position."""
        row, col = np.random.randint(0, self.grid_size, size=2)
        self.current_target = (row, col)
        self.target_set.emit(row, col)

    def update_cursor_position(self, x: float, y: float):
        """Updates cursor position and checks if it is within the target."""
        row, col = self.current_target
        dx = (self._x - x) / self.grid_size
        dy = (self._y - y) / self.grid_size
        self._delta = (dx ** 2 + dy ** 2) * 0.75 + self._delta * 0.25
        self._x = x
        self._y = y

        if int(x) == col and int(y) == row:
            if not self._is_cursor_in_target:
                self._hold_timer.start()  # Start hold timer when entering target
                self._is_cursor_in_target = True
        else:
            if self._is_cursor_in_target:
                self._hold_timer.stop()  # Stop hold timer when exiting target
                self._is_cursor_in_target = False

        if self._at_rest and (self._delta > self._delta_threshold):
            self._at_rest = False
            self.moving.emit()
            if self._move_start_ms is None:
                self._move_start_ms = time.time_ns() / 1_000_000
        elif not self._at_rest and (self._delta < self._delta_threshold):
            self._at_rest = True
            self.resting.emit()
            if self._is_cursor_in_target:
                self.trial_duration[self._current_trial] = time.time_ns() / 1_000_000 - self._move_start_ms

    def on_target_hold_complete(self):
        """Handles target hold completion and selects a new target."""
        self.target_hold_completed.emit()
        self._current_trial += 1
        if self._current_trial == self._total_trials:
            self.stop_task()
        else:
            self.select_new_target()
            self._move_start_ms = None
            self._at_rest = True
    

class FittsTargetWindow(GuiWindow):
    new_target = pyqtSignal(int, int)
    calibration_state = pyqtSignal(bool)

    _fsm: FittsGridStateMachine = None

    _grid_size: int = 5

    row_spin: QtWidgets.QSpinBox = None
    col_spin: QtWidgets.QSpinBox = None
    send_target_button: QtWidgets.QPushButton = None
    calibration_checkbox: QtWidgets.QCheckBox = None

    def __init__(self, fsm: FittsGridStateMachine):
        super().__init__(set_layout=False)
        self._fsm = fsm
        self.setWindowTitle("Target Info")
        self.setWindowIcon(QtGui.QIcon(os.path.join(self._assets, "FittsTargetIcon.png")))
        self.setGeometry(1000, 1000, 350, 100)
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)

        self.row_spin = QtWidgets.QSpinBox()
        self.row_spin.setRange(1, 5)
        self.row_spin.setValue(2)
        self.layout.addWidget(QtWidgets.QLabel("Row"), 0, 0, 1, 1)
        self.layout.addWidget(self.row_spin, 1, 0, 1, 1)

        self.col_spin = QtWidgets.QSpinBox()
        self.col_spin.setRange(1, 5)
        self.col_spin.setValue(3)
        self.layout.addWidget(QtWidgets.QLabel("Col"), 0, 1, 1, 1)
        self.layout.addWidget(self.col_spin, 1, 1, 1, 1)

        self.send_target_button = QtWidgets.QPushButton("Send Target")
        self.send_target_button.clicked.connect(self.handle_send_target_click)
        self.layout.addWidget(self.send_target_button, 0, 2, 1, 1)

        self.calibration_checkbox = QtWidgets.QCheckBox("Calibrate")
        self.calibration_checkbox.clicked.connect(self.calibration_state.emit)
        self.layout.addWidget(self.calibration_checkbox, 1, 2, 1, 1)
        self.show()

        self._fsm.target_set.connect(self.on_target_set)

    def handle_send_target_click(self):
        # Sending row/col as 0-indexed.
        row = self.row_spin.value()-1
        col = self.col_spin.value()-1
        self.new_target.emit(row, col)

    @pyqtSlot(int, int)
    def on_target_set(self, row: int, col: int):
        # Receive as 0-indexed; displays as 1-indexed.
        self._grid_size = max(max(self._grid_size, row+1),col+1)
        self.row_spin.setRange(1, self._grid_size)
        self.row_spin.setValue(row+1)
        self.col_spin.setRange(1, self._grid_size)
        self.col_spin.setValue(col+1)

    @pyqtSlot(int)
    def on_grid_size_change(self, grid_size: int):
        self._grid_size = grid_size
        self.row_spin.setValue(min(self.row_spin.value(), grid_size))
        self.row_spin.setRange(1, grid_size)
        self.col_spin.setValue(min(self.col_spin.value(), grid_size))
        self.col_spin.setRange(1, grid_size)
    

class FittsGridWindow(GuiWindow):
    fitts_event = pyqtSignal(float)
    base_gain_changed = pyqtSignal(float)
    omega_threshold_changed = pyqtSignal(object)
    coefficient_adaptation = pyqtSignal(object)
    
    task: FittsGridStateMachine = None
    layout: QtWidgets.QGridLayout = None
    x: float = 0.0
    _kx: float = 0.0
    y: float = 0.0
    _ky: float = 0.0
    error_model: "np.ndarray[np.float32]" = np.zeros((11,2)).astype(np.float32)
    _adaptation_counter: int = 0
    _adaptation_period: int = 200
    _adaptive: bool = False
    _delta_vectors = np.random.randn(9,2).astype(np.float32) * 0.0000025  # Small initial random deltas
    _cumulative_deltas = np.zeros((9,2)).astype(np.float32)
    _coefficients: "np.ndarray[np.float32]" = None
    _omega_threshold_base = np.array([0.02, 0.015]).astype(np.float32)
    _base_gain: float = 1.0
    _gain: float = 1.0
    _offset: float = np.pi * 0.11
    _base_offset: float = np.pi * 0.11
    _canvas: pg.PlotItem = None
    _xy: pg.ScatterPlotItem = None
    _resting: bool = True
    _indicator: Tuple[pg.PlotCurveItem, pg.PlotCurveItem, pg.PlotCurveItem, pg.PlotCurveItem] = None
    _error_buffer = []
    _collecting_pls_calibration_data: bool = False
    _use_keyboard: bool = False
    keyboard: Controller = Controller()
    _pressed_keys: set = set()  # Track currently held keys
    _momentum_history = deque(maxlen=5)  # Store recent (dx, dy) pairs
    _momentum_thresh = 0.05  # You can tune this
    
    def __init__(self, model = default_model):
        super().__init__(set_layout=False)
        self.setWindowTitle("Fitts' Grid Task")
        self.setWindowIcon(QtGui.QIcon(os.path.join(self._assets, "FittsGridWindowIcon.png")))
        self.layout = QtWidgets.QGridLayout(self)
        self._coefficients = model['coeff']
        self.setLayout(self.layout)
        self.grid_size = 5  # Default grid size
        self.total_trials = 20

        # Initialize state machine
        self.task = FittsGridStateMachine(grid_size=self.grid_size, num_trials=self.total_trials)
        self.task.target_set.connect(self.set_target)
        self.task.task_completed.connect(self.on_task_completed)
        self.task.target_hold_completed.connect(self.on_target_hold_completed)
        self.task.moving.connect(self.on_moving)
        self.task.resting.connect(self.on_resting)

        # Initialize Target-Setter window
        self.target_handler = FittsTargetWindow(fsm=self.task)

        # Plot for grid
        self._canvas = pg.PlotWidget()
        self.layout.addWidget(self._canvas, 2, 0, 5, 7)
        self._canvas.setXRange(0, self.grid_size)
        self._canvas.setYRange(0, self.grid_size)
        self._canvas.showGrid(x=True, y=True)

        # Target marker
        self._target_marker = pg.ScatterPlotItem(x=[2.5], y=[2.5], size=30, pen=pg.mkPen('red'), brush=pg.mkBrush('red'))
        self._canvas.addItem(self._target_marker)

        # Moving cursor (x, y)
        self._xy = pg.ScatterPlotItem(x=[0.0], y=[0.0], size=15, pen=pg.mkPen('blue'), brush=pg.mkBrush('blue'))
        self._canvas.addItem(self._xy)

        # Start/Stop and settings controls
        self.start_button = QtWidgets.QPushButton("Start Task")
        self.start_button.clicked.connect(self.start)
        self.layout.addWidget(self.start_button, 0, 0, 1, 1)

        self.stop_button = QtWidgets.QPushButton("Stop Task")
        self.stop_button.clicked.connect(self.task.stop_task)
        self.layout.addWidget(self.stop_button, 0, 1, 1, 1)

        self.gain_spin = QtWidgets.QSpinBox()
        self.gain_spin.setRange(1, 1000)
        self.gain_spin.setValue(100)
        self.gain_spin.valueChanged.connect(self.update_gain)
        self.layout.addWidget(QtWidgets.QLabel("Gain (%)"), 1, 0, 1, 1)
        self.layout.addWidget(self.gain_spin, 1, 1, 1, 1)

        # Grid size and trials controls
        self.grid_size_spin = QtWidgets.QSpinBox()
        self.grid_size_spin.setRange(2, 20)
        self.grid_size_spin.setValue(self.grid_size)
        self.grid_size_spin.valueChanged.connect(self.update_grid_size)
        self.grid_size_spin.valueChanged.connect(self.target_handler.on_grid_size_change)
        self.layout.addWidget(QtWidgets.QLabel("Size:"), 0, 2, 1, 1)
        self.layout.addWidget(self.grid_size_spin, 1, 2, 1, 1)

        self.trials_spin = QtWidgets.QSpinBox()
        self.trials_spin.setRange(0, 1000)
        self.trials_spin.setValue(self.total_trials)
        self.trials_spin.valueChanged.connect(self.update_total_trials)
        self.layout.addWidget(QtWidgets.QLabel("Trials:"), 0, 3, 1 ,1)
        self.layout.addWidget(self.trials_spin, 1, 3, 1 ,1)

        self.deadzone_spin_x = QtWidgets.QSpinBox()
        self.deadzone_spin_x.setRange(0, 1000)
        self.deadzone_spin_x.setValue(100)
        self.deadzone_spin_x.valueChanged.connect(self.update_deadzone)
        self.layout.addWidget(QtWidgets.QLabel("X-Deadzone (%)"), 0, 4, 1, 1)
        self.layout.addWidget(self.deadzone_spin_x, 1, 4, 1, 1)

        self.deadzone_spin_y = QtWidgets.QSpinBox()
        self.deadzone_spin_y.setRange(0, 1000)
        self.deadzone_spin_y.setValue(100)
        self.deadzone_spin_y.valueChanged.connect(self.update_deadzone)
        self.layout.addWidget(QtWidgets.QLabel("Y-Deadzone (%)"), 0, 5, 1, 1)
        self.layout.addWidget(self.deadzone_spin_y, 1, 5, 1, 1)

        self.adaptation_checkbox = QtWidgets.QCheckBox("Enable Adaptive")
        self.adaptation_checkbox.setChecked(self._adaptive)
        self.layout.addWidget(self.adaptation_checkbox, 0, 6, 1, 1)
        self.adaptation_checkbox.clicked.connect(self.set_adaptation_state)

        self.adaptation_spinbox = QtWidgets.QSpinBox()
        self.adaptation_spinbox.setRange(0, 100)
        self.adaptation_spinbox.setValue(100)
        self.layout.addWidget(self.adaptation_spinbox, 1, 6, 1, 1)

        self.keyboard_checkbox = QtWidgets.QCheckBox("Keyboard")
        self.keyboard_checkbox.setChecked(False)
        self.layout.addWidget(self.keyboard_checkbox, 0, 7, 1, 1)
        self.keyboard_checkbox.clicked.connect(self.set_keyboard_state)

        # Initialize boundary lines as thick red lines, initially invisible
        self._indicator = (
            pg.PlotCurveItem(pen=pg.mkPen('red', width=4), visible=False),  # Left boundary
            pg.PlotCurveItem(pen=pg.mkPen('red', width=4), visible=False),  # Top boundary
            pg.PlotCurveItem(pen=pg.mkPen('red', width=4), visible=False),  # Right boundary
            pg.PlotCurveItem(pen=pg.mkPen('red', width=4), visible=False)   # Bottom boundary
        )

        # Add lines to the canvas
        for line in self._indicator:
            self._canvas.addItem(line)

        self.update_grid_size(self.grid_size)
        self.target_handler.new_target.connect(self.set_target)
        self.target_handler.calibration_state.connect(self.on_toggle_pls_calibration)

    def set_keyboard_state(self, enable: bool):
        """Sets the state of keyboard input for the task."""
        self._use_keyboard = enable

    def showEvent(self, event):
        event.accept()
        if self.target_handler is not None:
            self.target_handler.show()
            self.target_handler.raise_()

    def start(self):
        self.trials_spin.setEnabled(False)
        self.task.start_task(self.x, self.y)
        self.fitts_event.emit(50.0)

    def on_moving(self):
        self._resting = False
        self.fitts_event.emit(52.0)

    def on_resting(self):
        self._resting = True
        self.fitts_event.emit(53.0)

    def on_target_hold_completed(self):
        self.fitts_event.emit(54.0)
        self.trials_spin.setValue(self.trials_spin.value()-1)

    def set_adaptation_state(self, enabled: bool):
        self._adaptive = enabled

    @pyqtSlot(int, int)
    def set_target(self, row: int, col: int):
        self._target_marker.setData([col + 0.5], [row + 0.5])  # Center in cell
        self.fitts_event.emit(51.0)

    def update_deadzone(self):
        x_thresh = self._omega_threshold_base[0] * (float(self.deadzone_spin_x.value()) / 100.0)
        y_thresh = self._omega_threshold_base[1] * (float(self.deadzone_spin_x.value()) / 100.0)
        self.omega_threshold_changed.emit(np.array([x_thresh, y_thresh]).astype(np.float32))

    def update_gain(self, gain_pct):
        self._base_gain = float(gain_pct) / 100.0
        self._offset = self._base_offset / self._base_gain
        self._gain = self.grid_size / (2 * self._offset) # Based on max/min values allowed for `omega` (rotation)
        self.base_gain_changed.emit(self._base_gain)

    def update_grid_size(self, size):
        self.grid_size = size
        self._gain = self._base_gain * size / (2 * self._offset) # Based on max/min values allowed for `omega` (rotation)
        self._canvas.setXRange(0, self.grid_size)
        self._canvas.setYRange(0, self.grid_size)
        self.task.grid_size = size
        self.task.entropy = math.log2(2 ** (2 * size)) # bits
        self._indicator[0].setData([0, 0], [0, self.grid_size])  # Left boundary
        self._indicator[1].setData([0, self.grid_size], [self.grid_size, self.grid_size])  # Top boundary
        self._indicator[2].setData([self.grid_size, self.grid_size], [0, self.grid_size])  # Right boundary
        self._indicator[3].setData([0, self.grid_size], [0, 0])  # Bottom boundary

    def update_total_trials(self, trials):
        self.total_trials = trials
        self.task.set_total_trials(trials)

    def update_boundaries_visibility(self, x: float, y: float):
        """
        Update visibility of boundary lines based on the position of `self._xy`.
        If `self._xy` goes beyond any boundary, show the corresponding red line.
        """
        # Check and update visibility for each boundary
        self._indicator[0].setVisible(x < 0)  # Left boundary
        self._indicator[1].setVisible(y > self.grid_size)  # Top boundary
        self._indicator[2].setVisible(x > self.grid_size)  # Right boundary
        self._indicator[3].setVisible(y < 0)  # Bottom boundary

    def update_error_buffer(self, x_error, y_error, dx, dy, envelope_data):
        """
        Stores historical movement and coefficient data for directional trend analysis.
        """

        # Append data to the buffer, storing the target direction, movement vector, and coefficients
        for sample in envelope_data.T:
            self._error_buffer.append({
                "x_error": x_error,
                "y_error": y_error, 
                "dx": dx, 
                "dy": dy, 
                "envelope": sample[1:].tolist()
            })

    @pyqtSlot(bool)
    def on_toggle_pls_calibration(self, enable):
        if enable:
            self._error_buffer = []
            self._collecting_pls_calibration_data = True
        else:
            self._collecting_pls_calibration_data = False
            self.error_model = ModelInteractor.perform_pls_error_regression(self._error_buffer)

    @pyqtSlot(object, float)
    def on_task_completed(self, durations: "np.ndarray[np.float32]", entropy: float):
        mean_trial_duration = np.mean(durations) / 1000.0
        average_bit_rate = entropy / mean_trial_duration
        self.fitts_event.emit(55.0)
        self.trials_spin.setEnabled(True) # Resets trial counter spinbox so we can set total trials once again.
        self.trials_spin.setValue(self.total_trials)
        s = f"The Fitts' Grid Task is complete. Bit-Rate ~ {average_bit_rate:.2f} bits/sec"
        QtWidgets.QMessageBox.information(self, "Task Completed", s)
        print(s)
        print("Trial Durations (ms):")
        print(durations)

    @pyqtSlot(object, object, object, object, bool)
    def update(self, omega: "np.ndarray[np.float32]", emg_envelopes: "np.ndarray[np.float32]", omega_threshold: "np.ndarray[np.float32]", omega_gain: "np.ndarray[np.float32]", omega_from_rates: bool):
        """Updates the cursor position and checks for target acquisition."""
        # Decode and scale cursor position based on omega values
        x, y = (omega[0] + self._offset) * self._gain, (omega[1] + self._offset) * self._gain
        dx = self.x - x
        dy = self.y - y
        xy_delta_errors = np.hstack((emg_envelopes.T,np.full((emg_envelopes.shape[1],1),dx,dtype=np.float32),np.full((emg_envelopes.shape[1],1),dy,dtype=np.float32))) @ self.error_model
        err = np.sum(xy_delta_errors, axis=0)
        dx_adj = 0.05 * err[0]
        dy_adj = 0.05 * err[1]

        # Apply asymmetric gains
        if dx_adj > 0:
            dx_adj = dx_adj * 2.5
        if dy_adj > 0:
            dy_adj = dy_adj * 50.0
        
        self.update_boundaries_visibility(x + dx_adj, y + dx_adj)
        self._xy.setData([x + dx_adj], [y + dy_adj])  # Update cursor position on the plot
        self.x = min(max(x + dx_adj,0),self.grid_size)
        self.y = min(max(y + dy_adj,0),self.grid_size)
        # dkx = self._kx - self.x
        # dky = self._ky - self.y 
        # self._momentum_history.append((dkx, dky))
        # alpha = 0.9
        # self._kx =  (alpha * self._kx + (1 - alpha) * x)
        # self._ky =  (alpha * self._ky + (1 - alpha) * y)

        if self._use_keyboard:
            c = self.grid_size / 2
            deadzone = 0.85 * c
            keys_to_press = set()

            if self.x > (c+deadzone):
                keys_to_press.add(Key.right)
            elif self.x < (c-deadzone):
                keys_to_press.add(Key.left)

            if self.y > (c+deadzone):
                keys_to_press.add(Key.up)
            elif self.y < (c-deadzone):
                keys_to_press.add(Key.down)

            # Release keys no longer needed
            keys_to_release = self._pressed_keys - keys_to_press
            for key in keys_to_release:
                self.keyboard.release(key)

            # Press new keys
            keys_to_add = keys_to_press - self._pressed_keys
            for key in keys_to_add:
                self.keyboard.press(key)

            # Update pressed key state
            self._pressed_keys = keys_to_press


        # Exit early if the task is not running
        if not self.task.running:
            return

        # Update the task's cursor position for target checking and feedback
        self.task.update_cursor_position(x + dx_adj, y + dy_adj)

        # Change target marker color if hovering over target
        if int(x) == self.task.current_target[1] and int(y) == self.task.current_target[0]:
            self._target_marker.setBrush(pg.mkBrush("green"))  # Target hover indication
            in_target = True
        else:
            self._target_marker.setBrush(pg.mkBrush("red"))  # Default target color
            in_target = False

        if not in_target and not omega_from_rates:
            # Calculate the target direction vector in the `x, y` space
            target_x, target_y = self.task.current_target[1] + 0.5, self.task.current_target[0] + 0.5
            target_direction = np.array([target_x - x, target_y - y])
            target_magnitude = np.linalg.norm(target_direction)

            if self._collecting_pls_calibration_data:
                x_error = target_direction[0]*0.1 - dx
                y_error = target_direction[1]*0.1 - dy
                self.update_error_buffer(x_error, y_error, dx, dy, emg_envelopes)

            # If adaptive mode is enabled and task is running
            if self._adaptive:
                # Normalize the target direction for error projection
                if target_magnitude > 0:
                    target_unit_direction = target_direction / target_magnitude
                else:
                    target_unit_direction = np.zeros_like(target_direction)

                # Project movement (dx, dy) onto the target direction to estimate error
                movement_vector = np.array([dx, dy])
                movement_along_target = np.dot(movement_vector, target_unit_direction)

                a = self.adaptation_spinbox.value()
                learning_rate = a / 100.0  # Scale appropriately
                self._adaptation_counter = (self._adaptation_counter + 1) % self._adaptation_period
                # If enough iterations have passed, apply adaptive gradient descent updates
                if self._adaptation_counter == 0:
                    self.adaptation_spinbox.setValue(max(a - 1, 0))  # Gradually reduce adaptation over time

                # Calculate the effect of the current delta vectors on movement toward the target
                perturbed_coefficients = self._coefficients + self._delta_vectors
                perturbed_delta_omega = emg_envelopes.T @ perturbed_coefficients
                perturbed_delta_omega_summed = np.zeros((1,2))
                perturbed_delta_omega_summed[0,0] = np.sum(np.where(np.abs(perturbed_delta_omega[:,0]) > omega_threshold[0], perturbed_delta_omega[:,0],0)) * omega_gain[0]
                perturbed_delta_omega_summed[0,1] = np.sum(np.where(np.abs(perturbed_delta_omega[:,1]) > omega_threshold[1], perturbed_delta_omega[:,1],0)) * omega_gain[1]

                # Calculate movement direction with perturbed coefficients
                perturbed_movement = perturbed_delta_omega_summed @ target_unit_direction
                delta_effectiveness = perturbed_movement - movement_along_target
                self._coefficients += learning_rate * self._delta_vectors * delta_effectiveness
                self._delta_vectors = np.random.randn(9, 2).astype(np.float32) * 0.000001

                # print(f"Updated coefficients:\n{self._coefficients}")
                self.coefficient_adaptation.emit(self._coefficients)

        