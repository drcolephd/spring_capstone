from nml.gui_window import GuiWindow
from nml.driver.stimulator import MotorStimulator
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
import os, math, time
from typing import Tuple
    

class StimulatorWindow(GuiWindow):
    stim_event = pyqtSignal(bool, float)

    stimulator: MotorStimulator = None

    _start_pct: float = 0.0
    _end_pct: float = 0.25
    _stim_on_time: int = 10
    _stim_off_time: int = 30
    _n_pulses: int = 1000
    _carrier_freq_hz: int = 1000
    _timer_period: int = 2
    _stim_time_counter: int = 0
    _stim_pulse: int = 0
    _stim_state: bool = False # False == OFF | True == ON
    _stim_vals = None
    _stim_dir: bool = False

    layout: QtWidgets.QGridLayout = None
    num_spin: QtWidgets.QSpinBox = None
    freq_spin: QtWidgets.QSpinBox = None
    start_pct_spin: QtWidgets.QDoubleSpinBox = None
    end_pct_spin: QtWidgets.QDoubleSpinBox = None
    t_on_spin: QtWidgets.QSpinBox = None
    t_off_spin: QtWidgets.QSpinBox = None
    start_stop_button: QtWidgets.QPushButton = None
    counter_label: QtWidgets.QLabel = None

    timer: QTimer = None

    def __init__(self):
        super().__init__(set_layout=False)
        self.stimulator = MotorStimulator()
        self.setWindowTitle("Ramp Stimulator GUI")
        self.setWindowIcon(QtGui.QIcon(os.path.join(self._assets, "ModelInteractorIcon.png")))
        self.setGeometry(1000, 1000, 350, 100)
        self._initialize_cmu_style()
        self._initialize_widgets()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_stimulator)

    def _initialize_widgets(self):
        self.layout = QtWidgets.QGridLayout(self)
        self.setLayout(self.layout)
        self.num_spin = QtWidgets.QSpinBox()
        self.num_spin.setRange(1, 10000)
        self.num_spin.setValue(self._n_pulses)
        self.num_spin.valueChanged.connect(self.update_stim_parameters)
        self.layout.addWidget(QtWidgets.QLabel("N Pulses"), 0, 0, 1, 1)
        self.layout.addWidget(self.num_spin, 0, 1, 1, 1)

        self.freq_spin = QtWidgets.QSpinBox()
        self.freq_spin.setRange(32, 1028)
        self.freq_spin.setValue(self._carrier_freq_hz)
        self.freq_spin.valueChanged.connect(self.update_stim_parameters)
        self.layout.addWidget(QtWidgets.QLabel("Carrier (Hz)"), 0, 2, 1, 1)
        self.layout.addWidget(self.freq_spin, 0, 3, 1, 1)

        self.start_pct_spin = QtWidgets.QDoubleSpinBox()
        self.start_pct_spin.setRange(0.0, 1.0)
        self.start_pct_spin.setValue(self._start_pct)
        self.start_pct_spin.setDecimals(3)
        self.start_pct_spin.valueChanged.connect(self.update_stim_parameters)
        self.layout.addWidget(QtWidgets.QLabel("Start"), 1, 0, 1, 1)
        self.layout.addWidget(self.start_pct_spin, 1, 1, 1, 1)

        self.end_pct_spin = QtWidgets.QDoubleSpinBox()
        self.end_pct_spin.setRange(0.0, 1.0)
        self.end_pct_spin.setValue(self._end_pct)
        self.end_pct_spin.setDecimals(3)
        self.end_pct_spin.valueChanged.connect(self.update_stim_parameters)
        self.layout.addWidget(QtWidgets.QLabel("End"), 1, 2, 1, 1)
        self.layout.addWidget(self.end_pct_spin, 1, 3, 1, 1)

        self.t_on_spin = QtWidgets.QSpinBox()
        self.t_on_spin.setRange(5, 1000)
        self.t_on_spin.setValue(self._stim_on_time)
        self.t_on_spin.valueChanged.connect(self.update_stim_parameters)
        self.layout.addWidget(QtWidgets.QLabel("T HIGH (ms)"), 2, 0, 1, 1)
        self.layout.addWidget(self.t_on_spin, 2, 1, 1, 1)

        self.t_off_spin = QtWidgets.QSpinBox()
        self.t_off_spin.setRange(5, 1000)
        self.t_off_spin.setValue(self._stim_off_time)
        self.t_off_spin.valueChanged.connect(self.update_stim_parameters)
        self.layout.addWidget(QtWidgets.QLabel("T LOW (ms)"), 2, 2, 1, 1)
        self.layout.addWidget(self.t_off_spin, 2, 3, 1, 1)

        self.counter_label = QtWidgets.QLabel("N = 0")
        self.layout.addWidget(self.counter_label, 3, 0, 1, 1)

        self.start_stop_button = QtWidgets.QPushButton("Stimulate")
        self.start_stop_button.clicked.connect(self.handle_start_click)
        self.start_stop_button.setStyleSheet("QPushButton { background-color: yellow; color: black; border-style: outset; border-width: 2px; border-radius: 10px; border-color: beige; font: bold 14px; padding: 6px;}")
        self.layout.addWidget(self.start_stop_button, 3, 2, 1, 2)
        self.show()

    def __del__(self):
        self.stimulator.send_single_motor_command(0, 0.0, True)
        self.stimulator.set_carrier_frequency(100)
        self.stimulator.close()

    def update_stimulator(self):
        self._stim_time_counter += self._timer_period
        if self._stim_state:
            if self._stim_time_counter == self._stim_on_time:
                self._stim_state = False
                self.stim_event.emit(False, self._stim_pulse)
                self.stimulator.send_single_motor_command(0, 0.0, False, self._stim_dir)
                self._stim_dir = not self._stim_dir
                self._stim_pulse += 1
                self.counter_label.setText(f"N = {self._stim_pulse}")
                self._stim_time_counter = 0
        else:
            if self._stim_time_counter == self._stim_off_time:
                if self._stim_pulse < self._n_pulses:
                    self._stim_state = True
                    val = self._stim_vals[self._stim_pulse] # type: ignore
                    self.stim_event.emit(True, val)
                    self.stimulator.send_single_motor_command(0, val)
                    self._stim_time_counter = 0
                else:
                    self.handle_stop_click()

    def update_stim_parameters(self, _):
        self._stim_off_time = self.t_off_spin.value()
        self._stim_on_time = self.t_on_spin.value()
        self._start_pct = self.start_pct_spin.value()
        self._end_pct = self.end_pct_spin.value()
        self._n_pulses = self.num_spin.value()
        self._carrier_freq_hz = self.freq_spin.value()

    def handle_start_click(self):
        self.stimulator.set_carrier_frequency(self._carrier_freq_hz)
        self._stim_vals = np.linspace(self._start_pct, self._end_pct, self._n_pulses)
        self.timer.start(self._timer_period)
        self.start_stop_button.setStyleSheet("QPushButton { background-color: red; color: white; border-style: outset; border-width: 2px; border-radius: 10px; border-color: beige; font: bold 14px; padding: 6px;}")

        self.start_stop_button.setText("Stop")
        self.start_stop_button.clicked.disconnect(self.handle_start_click)
        self.start_stop_button.clicked.connect(self.handle_stop_click)

    def handle_stop_click(self):
        self.timer.stop()
        self.stimulator.send_single_motor_command(0, 0.0, True)
        self.stimulator.set_carrier_frequency(100)
        self._stim_state = False
        self._stim_time_counter = 0
        self._stim_pulse = 0
        self.start_stop_button.setStyleSheet("QPushButton { background-color: yellow; color: black; border-style: outset; border-width: 2px; border-radius: 10px; border-color: beige; font: bold 14px; padding: 6px;}")
        self.start_stop_button.setText("Stimulate")
        self.start_stop_button.clicked.disconnect(self.handle_stop_click)
        self.start_stop_button.clicked.connect(self.handle_start_click)
