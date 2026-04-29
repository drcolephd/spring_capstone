import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from pyqtgraph.Qt import QtCore, QtGui
from mindrove.board_shim import BoardShim, BoardIds, MindroveConfigMode
import numpy as np
from nml.processing import MontageMode, DEFAULT_ZCA_PARAMS
from nml.spikes import BeamformerSpikeHandler, NEOSpikeHandler
from nml.frequency_buffer import FrequencyBuffer
from nml.processor import Processor
from nml.direction_prompt_game import DirectionPromptGame
from nml.binary_logger import BinaryLogger
from nml.binary_reader import BinaryReader
from nml.local_paths import paths
from nml.feature_weights import weights
from nml.spike_scope import SpikeScope
from nml.clickable_viewbox import ClickableViewBox
from nml.marker_buttons_gui_panel import MarkerButtonsWindow
from nml.fitts_grid_window import FittsGridWindow
from nml.model_interactor import ModelInteractor
from nml.stimulator_window import StimulatorWindow
import keyboard, os
import matplotlib.cm as cm

PI = np.pi

class RealTimePlotter(QtWidgets.QWidget):
    closed = pyqtSignal() 
    stream_started = pyqtSignal()
    stream_stopped = pyqtSignal(int)
    block_changed = pyqtSignal(int)
    marker = pyqtSignal(float)
    mouse = None

    logger: BinaryLogger | None = None
    processor: Processor | None = None
    board_shim: BoardShim | None= None

    # Key parameters for online MUAP decomp!
    _pre_peak_samples = 6
    _post_peak_samples = 9
    _num_ipts = 8

    _colors = []
    _alpha = 0.98
    _threshold_fraction = 0.50
    _prompt = -1
    prompter = None
    features = np.zeros(4)
    _ulnar_magnitude = 1.0
    _radial_magnitude = 1.0
    _extensor_magnitude = 1.0
    buffer_size = 256 # samples in buffer
    graphics_refresh_timer = None
    _graphics_timer_period = 10 # milliseconds
    rest_calibration_timer = None
    direction_calibration_timer = None
    _calibration_timer_period = 500 # milliseconds
    _calibration_timer_start_delay = 2500 # milliseconds (to give time after clicking button, to rest or activate hand)
    _rest_buffer_capacity = 200
    _direction_buffer_capacity = 2000
    _debounce_interval = 450 # milliseconds
    _save_logs: bool = False
    _running = False
    _calibrating = False
    _has_baseline = False
    _has_threshold = True
    _has_calibration = False
    _has_scope = False
    _has_marker_window = False
    _has_auto_thresholds = False
    keyboard_checkbox = None
    _use_keyboard = False
    _arrow_on = False
    _arrow_direction = -1
    _arrow_angle = [90, 180, 270, 0]
    _arrow_position = [[0, 20], [25, 0], [0, -20], [-25, 0]]
    _arrow_key_map = ['up', 'right', 'down', 'left']
    _detection_threshold = 0.6
    _expected_signal_peak_amplitude = 250 # microvolts
    _max_plotted_frequency = 100 # Hz
    _orientation = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    start_button = None
    stop_button = None
    alpha_spinbox = None
    # ipts_threshold_spinbox = None
    marker_window = None
    markers_button = None
    fitts_grid_button = None
    load_button = None
    stim_button = None
    model_interactor_button = None
    record_button = None
    calibration_button_rest = None
    calibration_button_direction = None
    scope_button = None
    covariance_button = None
    muaps_button = None
    orientation_plot = None
    orientation_arrows = []
    decode_plot = None
    decode_arrow = None
    features_plot = None
    feature_bars = None
    feature_curves = None
    # base_angle = np.array([6.2, 4.9, 4.7, 4.5, 4.2, 2.0, 1.7, 1.4])  
    base_angle = np.linspace(2*PI,0,9)[:-1]
    angle = np.zeros(8)
    time_plots = []
    time_curves = []
    freq_plots = []
    freq_curves = []
    threshold_lines = []
    _emg_threshold = 1500
    _emg_channel = 0
    spike_scope: SpikeScope | None = None
    fitts_grid_window: FittsGridWindow | None = None
    model_interactor_window: ModelInteractor | None = None
    stimulator_window: StimulatorWindow | None = None

    def __init__(self, app, board_shim, is_main_window: bool = True, save_logs: bool = False, filename: str = None, suffix: int = None, buffer_size: int = None):
        super().__init__()
        self.board_shim = board_shim
        self.is_main_window = is_main_window
        self.board_id = self.board_shim.get_board_id()  # Get the board ID
        self.sampling_rate = self.board_shim.get_sampling_rate(self.board_id)
        self.exg_channels = self.board_shim.get_exg_channels(self.board_id)
        self.accel_channels = self.board_shim.get_accel_channels(self.board_id)
        self.gyro_channels = self.board_shim.get_gyro_channels(self.board_id)
        self.app = app
        self.filename = filename
        self.suffix = suffix
        self._save_logs = save_logs
        if buffer_size is not None:
            self.buffer_size = buffer_size
        self.fft_freqs = np.fft.fftfreq(self.buffer_size, d=1/self.sampling_rate)[:self.buffer_size // 2]
        self.mask = self.fft_freqs <= self._max_plotted_frequency
        self.adjusted_magnitude = np.zeros((8, len(self.fft_freqs[self.mask])))
        self.adjusted_mask = (self.fft_freqs[self.mask] >= 32) & (self.fft_freqs[self.mask] <= 64) 
        self.norm_factor = self.buffer_size * self._expected_signal_peak_amplitude
        if self.board_id == BoardIds.SYNTHETIC_BOARD:
            self.num_channels = 32
            print(f"Sampling at {self.sampling_rate} samples/sec.")
        else:
            self.num_channels = 35
        self.calibration_buffer = FrequencyBuffer(
                                    num_channels=len(self.exg_channels), 
                                    num_freq_bands=len(self.fft_freqs[self.mask]),  # pyright: ignore[reportCallIssue]
                                    capacity=self._rest_buffer_capacity,
                                    threshold_fraction=self._threshold_fraction) # Theoretically 200 * 10ms --> 2s for rest calibration buffer to fill
        self.processor = Processor(self.board_shim, self.buffer_size, 
                                   num_channels=self.num_channels, 
                                   filter_cutoff_hpf=100, # Hz 
                                   filter_cutoff_env=1, # Hz
                                   filter_order_hpf=1, 
                                   filter_order_env=1, 
                                   sample_rate=self.sampling_rate, 
                                   channel=self._emg_channel,
                                   threshold=self._emg_threshold, 
                                   pre_peak_samples=self._pre_peak_samples, 
                                   post_peak_samples=self._post_peak_samples, 
                                   num_ipts=self._num_ipts, 
                                   montage_mode=MontageMode.MONOPOLAR, 
                                   zca_params=DEFAULT_ZCA_PARAMS, 
                                   spike_handler_cls=BeamformerSpikeHandler)
        
        self.setGeometry(100, 100, 800, 800)
        self.setStyleSheet("background-color: black")
        self.setWindowIcon(QtGui.QIcon(os.path.join(paths['assets'], "MindRoveLogo.png")))
        self._update_window_title()
        # Arrange buttons and level in a vertical layout
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(5)
        self._initialize_start_stop_buttons(layout)
        self._initialize_keyboard_checkbox(layout)
        self._initialize_refmode_spinbox(layout)
        # self._initialize_alpha_spinbox(layout)
        # self._initialize_ipts_threshold_spinbox(layout)
        # self._initialize_model_loader_button(layout)
        self._initialize_stimulator_button(layout)
        self._initialize_markers_button(layout)
        self._initialize_model_interactor_button(layout)
        self._initialize_recording_button(layout)
        self._initialize_rates_model_checkbox(layout)
        self._initialize_fitts_grid_button(layout)
        self._initialize_calibration_buttons(layout)
        self._initialize_scope_button(layout)  # Initialize the Scope button
        self._initialize_muaps_button(layout)
        self._initialize_covariance_button(layout)
        self._initialize_zca_reset_button(layout)
        self.canvas = pg.GraphicsWindow()
        layout.addWidget(self.canvas, 4, 0, 12, 6)
        self.setLayout(layout)
        self._initialize_plots()
        self._initialize_timers()
        # Start the keyboard event handler
        if self.is_main_window:
            self.canvas.setFocus()
            self.canvas.grabKeyboard()
            self.canvas.installEventFilter(self)

    def setMouse(self, new_mouse):
        if new_mouse is None:
            pass
        else:
            self.mouse = new_mouse

    def closeEvent(self, event):
        """Handle the close event to ensure cleanup."""
        event.accept()  # Accept the close event
        if self._running:
            self.stop()
        self.closed.emit() # Emit closed event
        if self.is_main_window:
            self.app.quit()

    @pyqtSlot(float)
    def onCovarianceStateChange(self, newCovarianceState: float):
        self.board_shim.insert_marker(newCovarianceState + 60.0) # Covariance-0 = encoding 60.0

    def eventFilter(self, source, event):
        """Handle keyboard events."""
        if self._debounce_ok and self._running:
            if event.type() == QtCore.QEvent.KeyPress:
                if event.key() == QtCore.Qt.Key_Space:
                    print("Spacebar pressed: Sending 'beep' sync")
                    self.board_shim.config_board(MindroveConfigMode.BEEP)
                elif event.key() == QtCore.Qt.Key_Return:
                    print("Enter pressed: Sending 'boop' sync")
                    self.board_shim.config_board(MindroveConfigMode.BOOP)
                elif event.key() == QtCore.Qt.Key_Escape:
                    print("Escape pressed: exiting application")
                    self.close()
        return super().eventFilter(source, event)
    
    @pyqtSlot(object, bool)
    def on_new_manip_data(self, data: "np.ndarray[np.float32]", button_state: bool):
        self.processor.update_position(float(data[0]), float(data[1]), button_state)

    def update(self):
        """Update the plots."""
        if self._running:
            self._update_orientation_arrows()
            self._update_time_frequency_streams()
        self.app.processEvents()

    def start(self):
        """Start the QTimer only after initialization."""
        if self._running:
            print("Stream is already running!")
            return
        if self._save_logs:
            fname = f"{self.filename}_{self.suffix}.tsv"
            print(f"Logging streams to file: {fname}")
            self.board_shim.start_stream(self.buffer_size, f"file://{fname}:w")
            self.logger = BinaryLogger(f"{self.filename}_{self.suffix}.bin")
            self.suffix = self.suffix + 1
        else:
            self.board_shim.start_stream(self.buffer_size)
        self.processor.set_filename(self.filename, self.suffix)
        self.stream_started.emit()
        self._running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.calibration_button_rest.setEnabled(True)
        self.calibration_button_direction.setEnabled(self._has_threshold)
        self.processor.start_device_sampling() # timer to grab new samples from device
        self.processor.start_xy_interpolation()
        self.graphics_refresh_timer.start(self._graphics_timer_period)  # timer to update graphics

    def stop(self):
        """Stops the streams/timers."""
        if not self._running:
            return
        self._running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.calibration_button_rest.setEnabled(False)
        self.calibration_button_direction.setEnabled(False)
        self.processor.stop_xy_interpolation()  
        self.processor.stop_device_sampling()
        self.graphics_refresh_timer.stop()
        self.board_shim.stop_stream()
        if self.logger is not None:
            f = self.logger.filename
            self.logger.close()
            del self.logger
            self.logger = None
            r = BinaryReader(f)
            r.convert()
            r.close()
            del r
            self._update_window_title()
        if self.suffix is None:
            self.stream_stopped.emit(-1)
        else:
            self.stream_stopped.emit(self.suffix)

    @pyqtSlot(float)
    def on_prompt_marker_signal(self, val):
        self._prompt = val - 2

    @pyqtSlot(str, int, bool)
    def on_file_handling_update(self, new_fname, new_suffix, saving_to_file):
        if self._running:
            self.stop()
        self._save_logs = saving_to_file
        self.filename = new_fname
        self.suffix = new_suffix
        self._update_window_title()

    def _update_window_title(self):
        self.setWindowTitle(f"Real-Time Streams: {self.filename}_{self.suffix}")    

    @pyqtSlot()
    def handle_keyboard_checkbox_click(self):
        self._use_keyboard = self.keyboard_checkbox.isChecked() # type: ignore

    def _handle_auto_threshold_calculation(self):
        # self.processor.spikes.compute_auto_thresholds()
        self._has_auto_thresholds = True
        self.threshold_lines[self.processor.spikes._spike_channel].setVisible(True)
        self.threshold_lines[self.processor.spikes._spike_channel].setPos(self.processor.spikes._spike_threshold[self.processor.spikes._spike_channel])
        # if self.spike_scope is not None:
        #     self.spike_scope.set_spike_buffering_state(True)
        #     self.spike_scope.set_covariance_buffering_state(True)

    @pyqtSlot()
    def handle_rest_calibration_click(self):
        self.calibration_button_rest.setEnabled(False)
        self.calibration_buffer.set_capacity(self._rest_buffer_capacity)
        QtCore.QTimer.singleShot(self._calibration_timer_start_delay, self._handle_rest_calibration)
        self.marker.emit(40.0)

    def _handle_rest_calibration(self):
        self._calibrating = True
        self._has_calibration = False
        self.features = np.zeros(8)
        self.marker.emit(41.0)
        self.rest_calibration_timer.start(self._calibration_timer_period) # Check every half-second if buffer is full

    def end_rest_calibration(self):
        if self.calibration_buffer.is_full():
            self.rest_calibration_timer.stop()
            self._calibrating = False
            self.calibration_buffer.estimate_baseline()
            self._has_baseline = True
            self.processor.set_orientation()
            self._has_calibration = self._has_baseline and self._has_threshold
            self.calibration_button_rest.setEnabled(True)
            self.calibration_button_direction.setEnabled(True)
            self.calibration_button_rest.setStyleSheet("""
                QPushButton:enabled {
                    background-color: white;
                    border-color: black;
                    color: black;
                    font-size: 16px;
                }
                QPushButton:disabled {
                    background-color: lightgray;
                    color: gray;
                }
            """)
            self.marker.emit(42.0)
        else:
            print("Baseline-calibration (REST) buffer not yet full.")

    @pyqtSlot()
    def handle_direction_calibration_click(self):
        self.calibration_button_direction.setEnabled(False)
        self.calibration_buffer.set_capacity(self._direction_buffer_capacity)
        QtCore.QTimer.singleShot(self._calibration_timer_start_delay, self._handle_direction_calibration)
        self.marker.emit(43.0)

    def _handle_direction_calibration(self):
        self._calibrating = True
        # self._has_calibration = False
        self.features = np.zeros(8)
        self.prompter = DirectionPromptGame(add_buttons = False)
        self.prompter.marker.connect(self.on_prompt_marker_signal)
        self.prompter.show()
        self.prompter.start()
        self.marker.emit(44.0)
        self.direction_calibration_timer.start(self._calibration_timer_period) # Check every half-second if buffer is full

    def end_direction_calibration(self):
        if self.calibration_buffer.is_full():
            self.direction_calibration_timer.stop()
            self.prompter.stop()
            self.prompter.close()
            self.prompter = None
            self._prompt = -1
            self._calibrating = False
            self._has_threshold = True
            self._has_calibration = self._has_baseline and self._has_threshold
            self.calibration_buffer.estimate_threshold()
            features_filename = f"{self.filename}_{self.suffix}_features.csv"
            self.calibration_buffer.export_features(features_filename)
            self.calibration_button_direction.setEnabled(True)
            self.calibration_button_direction.setStyleSheet("""
                QPushButton:enabled {
                    background-color: white;
                    border-color: black;
                    color: black;
                    font-size: 16px;
                }
                QPushButton:disabled {
                    background-color: lightgray;
                    color: gray;
                }
            """)
            self.marker.emit(45.0)
        else:
            print("Threshold-calibration (ACTIVE) buffer not yet full.")

    @pyqtSlot()
    def handle_start_click(self):
        print("Stream started")
        self.start()

    @pyqtSlot()
    def handle_stop_click(self):
        print("Stream stopped")
        self.stop()

    @pyqtSlot()
    def handle_alpha_value_change(self):
        self._alpha = float(self.alpha_spinbox.value()) / 100.0
        self.processor.set_alpha(self._alpha)

    @pyqtSlot()
    def handle_detection_threshold_value_change(self):
        self._threshold_fraction = float(self.threshold_spinbox.value()) / 100.0
        self.calibration_buffer.set_threshold_fraction(self._threshold_fraction)
        if self._has_threshold:
            self.calibration_buffer.estimate_threshold() # Recalculate the threshold based on updated fraction

    def _reset_debounce_flag(self):
        self._debounce_ok = True

    def _initialize_zca_reset_button(self, layout):
        """
        Add a button that forces the processor's ZCA whitening to reset and re-train.
        """
        self.zca_reset_button = QtWidgets.QPushButton("Reset Whitening", self)
        self.zca_reset_button.setStyleSheet("""
            QPushButton:enabled {
                background-color: white;
                color: purple;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.zca_reset_button.clicked.connect(self._handle_zca_reset_click)
        
        # Add it to the grid (row 3 as an example)
        layout.addWidget(self.zca_reset_button, 3, 0, 1, 1)

    @pyqtSlot()
    def _handle_zca_reset_click(self):
        if self.processor is not None:
            print("[ZCA] Reset requested by user.")
            self.processor._zca.reset()

            # Optional: emit a marker so your streams label the reset
            try:
                self.marker.emit(99.0)   # marker code for ZCA reset
            except Exception:
                pass

    def _initialize_markers_button(self, layout):
        self.markers_button = QtWidgets.QPushButton("Markers", self)
        self.markers_button.setStyleSheet("""
            QPushButton:enabled {
                background-color: white;
                color: black;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.markers_button.clicked.connect(self._show_marker_window)
        layout.addWidget(self.markers_button, 1, 0)

    def _show_marker_window(self):
        if not self._has_marker_window:
            self.marker_window = MarkerButtonsWindow()
            self.marker_window.markerClicked.connect(self.marker.emit)
            self.marker_window.closed.connect(self._on_marker_window_close)
            self.closed.connect(self.marker_window.handleParentClosing)
            self._has_marker_window = True
        self.marker_window.show()
        self.marker_window.raise_()

    def _on_marker_window_close(self):
        self._has_marker_window = False
        del self.marker_window
        self.marker_window = None

    def _initialize_start_stop_buttons(self, layout):
        # Create Start and Stop buttons
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setStyleSheet("""
            QPushButton:enabled {
                background-color: white;
                color: blue;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setStyleSheet("""
            QPushButton:enabled {
                background-color: white;
                color: red;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        # Connect button signals
        self.start_button.clicked.connect(self.handle_start_click)
        self.stop_button.clicked.connect(self.handle_stop_click)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button, 0, 0, 1, 1)
        layout.addWidget(self.stop_button, 0, 1, 1, 1)

    def _initialize_model_interactor_button(self, layout):
        self.model_interactor_button = QtWidgets.QPushButton("Open Model", self)
        layout.addWidget(self.model_interactor_button, 1, 1, 1, 1)
        self.model_interactor_button.setStyleSheet("""
            QPushButton:enabled {
                font-size: 16px;
                font-weight: bold; 
                background-color: white;
                color: blue;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.model_interactor_button.clicked.connect(self.handle_model_interactor_button_click)

    def _initialize_recording_button(self, layout):
        self.record_button = QtWidgets.QPushButton("Start EMG Recording", self)
        layout.addWidget(self.record_button, 1, 2, 1, 2)
        self.record_button.setStyleSheet("""
            QPushButton:enabled {
                font-size: 16px;
                font-weight: bold; 
                background-color: white;
                color: green;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.record_button.clicked.connect(self.handle_start_recording_click)

    def handle_start_recording_click(self):
        fname = f"{self.filename}_{self.suffix}_emg_xy.bin"
        fname_spikes = f"{self.filename}_{self.suffix}_spikes_xy.bin"
        self.processor.start_emg_only_recording(fname, fname_spikes)
        self.record_button.setText("Stop EMG Recording")
        self.record_button.setStyleSheet("""
            QPushButton:enabled {
                font-size: 16px;
                font-weight: bold; 
                background-color: red;
                color: white;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.record_button.clicked.disconnect(self.handle_start_recording_click)
        self.record_button.clicked.connect(self.handle_stop_recording_click)

    def handle_stop_recording_click(self):
        self.processor.stop_emg_only_recording()
        self.record_button.setText("Start EMG Recording")
        self.record_button.setStyleSheet("""
            QPushButton:enabled {
                font-size: 16px;
                font-weight: bold; 
                background-color: white;
                color: green;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.record_button.clicked.disconnect(self.handle_stop_recording_click)
        self.record_button.clicked.connect(self.handle_start_recording_click)
        self.suffix = self.suffix + 1
        self.block_changed.emit(self.suffix)

    def _initialize_rates_model_checkbox(self, layout):
        self.rates_model_checkbox = QtWidgets.QCheckBox()
        self.rates_model_checkbox.setChecked(False)
        self.rates_model_checkbox.setText("Calibrate Rates")
        self.rates_model_checkbox.setStyleSheet("""
            QCheckBox {
                background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent for readability */
                color: black; 
                border: 1px solid #A50021; /* CMU red for subtle border */
                font-size: 16px;
                padding: 4px;
                border-radius: 4px;
            }
            /* Hover effect for QCheckBox */
            QCheckBox:hover {
                color: #A50021; /* Change checkbox text color on hover to CMU red */
            }
            QCheckBox::indicator:hover {
                background-color: #A50021; /* Highlight checkbox indicator on hover */
                border: 1px solid #4D4D4D; /* Darker border */
            }
        """)
        layout.addWidget(self.rates_model_checkbox,1,4,1,1)
        self.rates_model_checkbox.clicked.connect(self.processor.handle_rates_model_checkbox_click)

    def _initialize_fitts_grid_button(self, layout):
        self.processor.set_mode(1) # Ensures we are in the correct decode mode for this task.
        self.fitts_grid_button = QtWidgets.QPushButton("Fitts' Grid Task")
        self.fitts_grid_button.clicked.connect(self.handle_start_fitts_grid_window)
        layout.addWidget(self.fitts_grid_button, 1, 5, 1, 1)
        self.fitts_grid_button.setStyleSheet("""
            QPushButton:enabled {
                font-size: 16px;
                font-weight: bold; 
                background-color: white;
                color: green;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)

    def handle_model_interactor_button_click(self):
        if self.model_interactor_window is None:
            self.model_interactor_window = ModelInteractor()
            self.model_interactor_window.model_update.connect(self.processor.on_model_update)
            self.model_interactor_window.closed.connect(self.on_model_interactor_window_close)
            if self.fitts_grid_window is not None:
                self.fitts_grid_window.coefficient_adaptation.connect(self.model_interactor_window.on_coefficient_adaptation)
        self.model_interactor_window.show()
        self.model_interactor_window.raise_()
        self.model_interactor_button.setEnabled(False)

    @pyqtSlot()
    def on_model_interactor_window_close(self):
        self.model_interactor_button.setEnabled(True)

    def handle_start_fitts_grid_window(self):
        if self.fitts_grid_window is None:
            self.fitts_grid_window = FittsGridWindow()
            self.fitts_grid_window.fitts_event.connect(self.on_fitts_event)
            self.fitts_grid_window.base_gain_changed.connect(self.processor.update_omega_limit)
            self.fitts_grid_window.omega_threshold_changed.connect(self.processor.update_omega_deadzone)
            self.processor.omega.connect(self.fitts_grid_window.update)
            self.fitts_grid_window.closed.connect(self._on_fitts_grid_window_closed)
            self.fitts_grid_window.coefficient_adaptation.connect(self.processor.on_model_update)
            if self.model_interactor_window is not None:
                self.fitts_grid_window.coefficient_adaptation.connect(self.model_interactor_window.on_coefficient_adaptation)
        self.fitts_grid_window.show()
        self.fitts_grid_window.raise_()
        self.fitts_grid_button.setEnabled(False)

    @pyqtSlot(float)
    def on_fitts_event(self, val):
        self.marker.emit(val)

    def _on_fitts_grid_window_closed(self):
        self.fitts_grid_button.setEnabled(True)

    def _initialize_keyboard_checkbox(self,layout):
        self.keyboard_checkbox = QtWidgets.QCheckBox()
        self.keyboard_checkbox.setChecked(False)
        self.keyboard_checkbox.setText("Use Keyboard")
        self.keyboard_checkbox.setStyleSheet("""
            QCheckBox {
                background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent for readability */
                color: black; 
                border: 1px solid #A50021; /* CMU red for subtle border */
                font-size: 16px;
                padding: 4px;
                border-radius: 4px;
            }
            /* Hover effect for QCheckBox */
            QCheckBox:hover {
                color: #A50021; /* Change checkbox text color on hover to CMU red */
            }
            QCheckBox::indicator:hover {
                background-color: #A50021; /* Highlight checkbox indicator on hover */
                border: 1px solid #4D4D4D; /* Darker border */
            }
        """)
        layout.addWidget(self.keyboard_checkbox,0,2,1,1)
        self.keyboard_checkbox.clicked.connect(self.handle_keyboard_checkbox_click)

    def _initialize_refmode_spinbox(self,layout):
        lab = QtWidgets.QLabel("Spatial Ref:")
        lab.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)  # Right-align the text vertically centered
        lab.setStyleSheet("""
            background-color: black;
            color: white;
            font-size: 16px; 
        """)
        layout.addWidget(lab, 0, 3, 1, 1)
        self.refmode_spinbox = QtWidgets.QSpinBox()
        self.refmode_spinbox.setMinimum(0)
        self.refmode_spinbox.setMaximum(2)
        self.refmode_spinbox.setValue(2)
        self.refmode_spinbox.setSingleStep(1)
        self.refmode_spinbox.setStyleSheet("""
            QSpinBox:enabled {
                background-color: white;
                color: black;
            }
            QSpinBox:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.refmode_spinbox.valueChanged.connect(self.processor.set_montage)
        layout.addWidget(self.refmode_spinbox, 0, 4, 1, 1)

    # def _initialize_alpha_spinbox(self,layout):
    #     lab = QtWidgets.QLabel("α:")
    #     lab.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)  # Right-align the text vertically centered
    #     lab.setStyleSheet("""
    #         background-color: black;
    #         color: white;
    #         font-size: 16px; 
    #     """)
    #     layout.addWidget(lab, 0, 3, 1, 1)
    #     self.alpha_spinbox = QtWidgets.QSpinBox()
    #     self.alpha_spinbox.setMinimum(0)
    #     self.alpha_spinbox.setMaximum(100)
    #     self.alpha_spinbox.setValue(int(round(self._alpha * 100.0)))
    #     self.alpha_spinbox.setSingleStep(1)
    #     self.alpha_spinbox.setStyleSheet("""
    #         QSpinBox:enabled {
    #             background-color: white;
    #             color: black;
    #         }
    #         QSpinBox:disabled {
    #             background-color: lightgray;
    #             color: gray;
    #         }
    #     """)
    #     self.alpha_spinbox.valueChanged.connect(self.handle_alpha_value_change)
    #     layout.addWidget(self.alpha_spinbox, 0, 4, 1, 1)

    # def _initialize_model_loader_button(self, layout):
    #     self.load_button = QtWidgets.QPushButton("Load Rates Model")
    #     self.load_button.setStyleSheet("""
    #         QPushButton:enabled {
    #             font-size: 16px;
    #             font-weight: bold; 
    #             background-color: white;
    #             color: green;
    #         }
    #         QPushButton:disabled {
    #             background-color: lightgray;
    #             color: gray;
    #         }
    #     """)
    #     layout.addWidget(self.load_button, 0, 5, 1, 1)
    #     self.load_button.clicked.connect(lambda _: self.processor.load_models())

    def _initialize_stimulator_button(self, layout):
        self.stim_button = QtWidgets.QPushButton("Stimulator")
        self.stim_button.setStyleSheet("""
            QPushButton:enabled {
                font-size: 16px;
                font-weight: bold; 
                background-color: white;
                color: green;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        layout.addWidget(self.stim_button, 0, 5, 1, 1)
        self.stim_button.clicked.connect(self._open_stimulator_window)

    def _open_stimulator_window(self):
        if self.stimulator_window is None:
            self.stimulator_window = StimulatorWindow()
            self.stimulator_window.stim_event.connect(self._handle_stim_event)
        else:
            self.stimulator_window.show()
    
    @pyqtSlot(bool, float)
    def _handle_stim_event(self, is_stimulating: bool, val: float):
        self.board_shim.insert_marker(val)

    # def _initialize_ipts_threshold_spinbox(self,layout):
    #     lab = QtWidgets.QLabel("IPTs Threshold")
    #     lab.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)  # Right-align the text vertically centered
    #     lab.setStyleSheet("""
    #         background-color: black;
    #         color: white;
    #         font-size: 16px; 
    #     """)
    #     layout.addWidget(lab, 0, 5, 1, 1)
    #     self.ipts_threshold_spinbox = QtWidgets.QSpinBox()
    #     self.ipts_threshold_spinbox.setMinimum(0)
    #     self.ipts_threshold_spinbox.setValue(25)
    #     self.ipts_threshold_spinbox.setMaximum(100000)
    #     self.ipts_threshold_spinbox.setSingleStep(1)
    #     self.ipts_threshold_spinbox.setStyleSheet("""
    #         QSpinBox:enabled {
    #             background-color: white;
    #             color: black;
    #         }
    #         QSpinBox:disabled {
    #             background-color: lightgray;
    #             color: gray;
    #         }
    #     """)
    #     self.ipts_threshold_spinbox.valueChanged.connect(self.handle_ipts_threshold_value_change)
    #     layout.addWidget(self.ipts_threshold_spinbox, 1, 5, 1, 1)

    def _initialize_calibration_buttons(self, layout):
        self.calibration_button_rest = QtWidgets.QPushButton("Calibrate Rest", self)
        self.calibration_button_rest.setEnabled(False)
        self.calibration_button_rest.clicked.connect(self.handle_rest_calibration_click)
        self.calibration_button_rest.setStyleSheet("""
            QPushButton:enabled {
                background-color: white;
                border-color: yellow;
                color: blue;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        layout.addWidget(self.calibration_button_rest, 2, 0, 1, 1)

        lab = QtWidgets.QLabel("Threshold Fraction:")
        lab.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)  # Right-align the text vertically centered
        lab.setStyleSheet("""
            background-color: black;
            color: white;
            font-size: 16px; 
        """)
        layout.addWidget(lab, 2, 1, 1, 1)
        self.threshold_spinbox = QtWidgets.QSpinBox()
        self.threshold_spinbox.setMinimum(0)
        self.threshold_spinbox.setMaximum(1000)
        self.threshold_spinbox.setValue(int(round(self._threshold_fraction * 100)))
        self.threshold_spinbox.setSingleStep(5)
        self.threshold_spinbox.setStyleSheet("""
            QSpinBox:enabled {
                background-color: white;
                color: black;
            }
            QSpinBox:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.threshold_spinbox.valueChanged.connect(self.handle_detection_threshold_value_change)
        layout.addWidget(self.threshold_spinbox, 2, 2, 1, 1)

        self.calibration_button_direction = QtWidgets.QPushButton("Calibrate Directions", self)
        self.calibration_button_direction.setEnabled(False)
        self.calibration_button_direction.clicked.connect(self.handle_direction_calibration_click)
        self.calibration_button_direction.setStyleSheet("""
            QPushButton:enabled {
                background-color: blue;
                border-color: yellow;
                color: white;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        layout.addWidget(self.calibration_button_direction, 2, 3, 1, 1)

    def _initialize_muaps_button(self, layout):
        """
        Initialize the MUAPS button and add it to the layout.
        """
        self.muaps_button = QtWidgets.QPushButton("Set MUAPs", self)
        self.muaps_button.setStyleSheet("""
            QPushButton:enabled {
                background-color: white;
                color: black;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.muaps_button.clicked.connect(self._handle_manual_clustering)  # Connect to the method that opens SpikeScope
        layout.addWidget(self.muaps_button, 2, 3, 1, 1)  # Add to row 3, column 4

    @pyqtSlot()
    def _handle_manual_clustering(self):
        self.processor.spikes.handle_manual_clustering()
        if self.spike_scope is not None:
            self.spike_scope.set_state_status(self.processor.spikes._state, self.processor.spikes._has_muaps)

    def _initialize_covariance_button(self, layout):
        """
        Initialize the Covariance button and add it to the layout.
        """
        self.covariance_button = QtWidgets.QPushButton("Set Covariance", self)
        self.covariance_button.setStyleSheet("""
            QPushButton:enabled {
                background-color: white;
                color: black;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.covariance_button.clicked.connect(self._handle_manual_covariance)  # Connect to the method that opens SpikeScope
        layout.addWidget(self.covariance_button, 2, 4, 1, 1)  # Add to row 3, column 5

    @pyqtSlot()
    def _handle_manual_covariance(self):
        self.processor.spikes.handle_manual_covariance_setter()
        if self.spike_scope is not None:
            self.spike_scope.set_state_status(self.processor.spikes._state, self.processor.spikes._has_muaps)


    def _initialize_scope_button(self, layout):
        """
        Initialize the Scope button and add it to the layout.
        """
        self.scope_button = QtWidgets.QPushButton("Scope", self)
        self.scope_button.setStyleSheet("""
            QPushButton:enabled {
                background-color: white;
                color: black;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """)
        self.scope_button.clicked.connect(self._open_scope)  # Connect to the method that opens SpikeScope
        layout.addWidget(self.scope_button, 2, 5, 1, 1)  # Add to row 3, column 6

    def _open_scope(self):
        """
        Opens a new SpikeScope window.
        """
        if not self._has_auto_thresholds: # Wait 2.5 seconds then compute auto-thresholds
            QtCore.QTimer.singleShot(2500, self._handle_auto_threshold_calculation)
        if not self._has_scope:
            self.spike_scope = SpikeScope(self.app, 
                                          self.processor.spikes, 
                                          self._colors, 
                                          sample_rate=self.sampling_rate, 
                                          pre_peak_samples=self._pre_peak_samples, 
                                          post_peak_samples=self._post_peak_samples, 
                                          num_ipts=self._num_ipts)
            self._has_scope = True
            self.spike_scope.closed.connect(self._handle_spike_scope_closing)
            self.spike_scope.rates.connect(self.processor.spikes.on_spike_scope_rates)
            self.spike_scope.covariance_state_change.connect(self.onCovarianceStateChange)
            self.closed.connect(self.spike_scope.handleParentClosing)
            self.processor.spike.connect(self.spike_scope.on_spike_signal)
            # self.processor.motor_units.connect(self.spike_scope.on_motor_unit_signal)
            self.processor.embedding.connect(self.spike_scope.on_embedding_signal)
            self.processor.cluster_color_assigned.connect(self.spike_scope.on_cluster_color_assigned)
            self.threshold_lines[self._emg_channel].setVisible(True)
            self.threshold_lines[self._emg_channel].setPos(self._emg_threshold)
        self.spike_scope.show()
        self.spike_scope.raise_()

    def _handle_spike_scope_closing(self):
        """
        Handle the closing of the SpikeScope window.
        """
        self._has_scope = False
        self.spike_scope = None
        for i, line in enumerate(self.threshold_lines):
            line.setVisible(False)

    def _initialize_timers(self):
        self.graphics_refresh_timer = QtCore.QTimer()
        self.graphics_refresh_timer.timeout.connect(self.update)
        self.rest_calibration_timer = QtCore.QTimer()
        self.rest_calibration_timer.timeout.connect(self.end_rest_calibration)
        self.direction_calibration_timer = QtCore.QTimer()
        self.direction_calibration_timer.timeout.connect(self.end_direction_calibration)
        self.debounce_timer = QtCore.QTimer()
        self.debounce_timer.setSingleShot(True)

    def _initialize_plots(self):
        """Initialize the time series plots for each channel."""
        self._initialize_colors()
        self._initialize_time_plots()
        self._initialize_frequency_plots()
        self._initialize_orientation_plot()
        self._initialize_feature_radial_plot()
        self._initialize_decode_plot()

    def _initialize_colors(self):
        # colors = [
        #     (0.1900, 0.2518, 0.6522),
        #     (0.2769, 0.4658, 0.9370),
        #     (0.1080, 0.8127, 0.8363),
        #     (0.3857, 0.9896, 0.4202),
        #     (0.8207, 0.9143, 0.2063),
        #     (0.9967, 0.6082, 0.1778),
        #     (0.8568, 0.2250, 0.1276),
        #     (0.6896, 0.1158, 0.0806)
        # ]
        # self._colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
        self._colors = RealTimePlotter.generateBaseColors("cool")

    def _initialize_feature_radial_plot(self):
        
        # Set up the plot for radial features
        self.features_plot = self.canvas.addPlot(col=4, row=8, title="Radial Features")
        self.features_plot.showAxis('left', False)
        self.features_plot.showAxis('bottom', False)
        # Create 8 PlotCurveItems for each radial direction
        self.feature_curves = [pg.PlotCurveItem(pen=pg.mkPen(color=self._colors[i], width=4)) for i in range(4)]
        for curve in self.feature_curves:
            self.features_plot.addItem(curve)
        # Fixed radial angles for 8 directions (in radians)
        self.features_plot.setXRange(-2, 2)
        self.features_plot.setYRange(-2, 2)

    def _initialize_time_plots(self):
        for i in range(8):
            vb = ClickableViewBox()
            time_plot = self.canvas.addPlot(row=i, col=0, rowspan=1, colspan=3, viewBox=vb)
            time_plot.showAxis('left', False)   
            time_plot.showAxis('bottom', i == 7)
            # time_plot.setYRange(-500, 500)
            if i == 0:
                time_plot.setTitle('Realtime EMG Data')
            # Create a horizontal line that will act as the threshold indicator
            threshold_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine))
            threshold_line.setVisible(False)  # Start hidden, only show on click
            time_plot.addItem(threshold_line)
            time_plot.setTitle(f"Ch-{i}", color=self._colors[i])

            self.time_plots.append(time_plot)
            time_curve = time_plot.plot(pen=pg.mkPen(color=self._colors[i], width=2))
            self.time_curves.append(time_curve)
            self.threshold_lines.append(threshold_line)
            # Connect the plot's click event to a callback function
            vb.plotClicked.connect(lambda event, plot_idx=i: self._on_time_plot_clicked(event, plot_idx))
    
    def _initialize_frequency_plots(self):
        for i in range(8):
            freq_plot = self.canvas.addPlot(row=i, col=4, rowspan=1, colspan=3)
            freq_plot.showAxis('left', False)
            freq_plot.showAxis('bottom', i == 7)
            # freq_plot.setYRange(-0.25, 1.0)
            if i == 0:
                freq_plot.setTitle('Frequency Content')
            freq_curve = freq_plot.plot(pen=pg.mkPen(color=self._colors[i], width=2))
            self.freq_plots.append(freq_plot)
            self.freq_curves.append(freq_curve)

    def _on_time_plot_clicked(self, event, plot_idx):
        """
        Callback for handling clicks on time plots. Shows/hides threshold lines
        and sets the threshold for spike detection.
        """        
        if not self._has_scope:
            return
        # Get the y-value of the click in the plot's coordinates
        mouse_point = self.time_plots[plot_idx].vb.mapSceneToView(event.scenePos())
        self._emg_threshold = mouse_point.y()
        self._emg_channel = plot_idx
        
        # self._emg_threshold = max(min(mouse_point.y(),2500.0),-2500.0)
        print(f"EMG-Threshold: Channel-{self._emg_channel} = {mouse_point.y()}")
        # Update all plots, showing the threshold line only on the clicked plot
        for i, line in enumerate(self.threshold_lines):
            if i == plot_idx:
                line.setVisible(True)
                line.setPos(self._emg_threshold)
            else:
                line.setVisible(False)
        self.processor.spikes.set_detector(channel=self._emg_channel, threshold=self._emg_threshold)
        self.spike_scope.set_title(channel=self._emg_channel, threshold=self._emg_threshold)

    def _initialize_orientation_plot(self):
        self.orientation_plot = self.canvas.addPlot(col=0, row=8, title="Wristband Orientation")
        self.orientation_plot.setXRange(-35, 35)
        self.orientation_plot.setYRange(-25, 25)
        self.orientation_plot.showAxis('left',False)
        self.orientation_plot.showAxis('bottom',False)
        self.orientation_arrows = {
            'x': pg.ArrowItem(angle=0, tipAngle=45, baseAngle=0, headLen=10, tailLen=40, tailWidth=4, brush='r'),
            'y': pg.ArrowItem(angle=0, tipAngle=45, baseAngle=0, headLen=10, tailLen=40, tailWidth=4, brush='g'),
            'z': pg.ArrowItem(angle=0, tipAngle=45, baseAngle=0, headLen=10, tailLen=40, tailWidth=4, brush='b')
        }
        for arrow in self.orientation_arrows.values():
            self.orientation_plot.addItem(arrow)
        self.orientation_arrows['x'].setPos(-20,0)
        self.orientation_arrows['y'].setPos(0, 0)
        self.orientation_arrows['z'].setPos(20, 0)

    def _initialize_decode_plot(self):
        '''
        Initialize plot with the single arrow indicating decoded output direction.
        '''
        self.decode_plot = self.canvas.addPlot(col=2, row=8, title="Decode")
        self.decode_plot.setXRange(-50, 50)
        self.decode_plot.setYRange(-50, 50)
        self.decode_plot.showAxis('left',False)
        self.decode_plot.showAxis('bottom',False)
        self.decode_arrow = pg.ArrowItem(angle=0, tipAngle=30, baseAngle=0, headLen=20, tailLen=20, tailWidth=4, brush='y')
        self.decode_plot.addItem(self.decode_arrow)
        self.decode_arrow.setPos(0, 0)
        self.decode_arrow.hide()

    def _handle_show_or_hide_arrow(self, new_direction):
        if not self.debounce_timer.isActive():
            arrow_on = new_direction > -1
            if arrow_on:
                if not self._arrow_on:
                    self.decode_arrow.show()
                    self._arrow_on = True
                    if not self._arrow_direction == new_direction:
                        self.decode_arrow.setRotation(self._arrow_angle[new_direction])
                        self.decode_arrow.setPos(self._arrow_position[new_direction][0], self._arrow_position[new_direction][1])
                        self._arrow_direction = new_direction 
                        if self._use_keyboard:
                            keyboard.press(self._arrow_key_map[new_direction])
                        self.debounce_timer.start(self._debounce_interval)
                else:
                    if not self._arrow_direction == new_direction:
                        self.decode_arrow.hide()
                        self._arrow_on = False
                        if self._use_keyboard:
                            keyboard.release(self._arrow_key_map[self._arrow_direction])
                        self.debounce_timer.start(self._debounce_interval * 3)
                        self._arrow_direction = -1
            elif (not arrow_on) and self._arrow_on:
                self.decode_arrow.hide()
                self._arrow_on = False
                if self._use_keyboard:
                    keyboard.release(self._arrow_key_map[self._arrow_direction])
                self.debounce_timer.start(self._debounce_interval * 3)
                self._arrow_direction = -1
            
    def _handle_direction_decode(self) -> int:
        # Calculate the weighted vector based on features and angles
        weighted_x = 0
        weighted_y = 0
        for i, feature_value in enumerate(self.features):
            weighted_x += feature_value * np.cos(self.angle[i])
            weighted_y += feature_value * np.sin(self.angle[i])

        # Compute the resulting magnitude and angle of the weighted vector
        resultant_magnitude = np.sqrt(weighted_x**2 + weighted_y**2)
        resultant_angle = np.arctan2(weighted_y, weighted_x)  # Angle in radians, between -pi and pi

        # Check if the resultant magnitude exceeds the threshold for any direction
        if self._arrow_on:
            threshold = 1.5 * np.mean(self.calibration_buffer.threshold)  # Use an overall threshold, or customize as needed
        else:
            threshold = 0.5 * np.mean(self.calibration_buffer.threshold)
        if resultant_magnitude < threshold:
            return -1  # At-rest position

        # Determine the direction based on the angle (quadrant)
        if -np.pi/4 <= resultant_angle < np.pi/4:
            return 1  # Right
        elif np.pi/4 <= resultant_angle < 3*np.pi/4:
            return 0  # Up
        elif -3*np.pi/4 <= resultant_angle < -np.pi/4:
            return 2  # Down
        else:
            return 3  # Left

    def _update_time_frequency_streams(self):
        data = self.processor.get()   
        for channel in range(8):
            self.time_curves[channel].setData(data[channel])
            fft_data = np.fft.fft(data[channel]/self.norm_factor)
            fft_magnitude = np.abs(fft_data)[:len(fft_data) // 2]  
            self.adjusted_magnitude[channel] = fft_magnitude[self.mask] - self.calibration_buffer.baseline[channel]
            self.freq_curves[channel].setData(self.fft_freqs[self.mask], self.adjusted_magnitude[channel])
            if self._has_baseline:
                self.features[channel] = np.trapz(np.where(self.adjusted_magnitude[channel,self.adjusted_mask] > 0, self.adjusted_magnitude[channel,self.adjusted_mask], 0))
            if self._calibrating:
                self.calibration_buffer.append(channel, self.adjusted_magnitude[channel])
                if channel==0:
                    self.calibration_buffer.add_label(self._prompt)
                    self.calibration_buffer.add_angle(self.angle)
        self._update_feature_radial_plot()
        if self._has_calibration:
            self._update_decode_arrow()

    def _update_orientation_arrows(self):
        orientation = self.processor.orientation
        self.orientation_arrows['x'].setRotation(np.rad2deg(orientation[0])+180)
        self.orientation_arrows['y'].setRotation(np.rad2deg(orientation[1]))
        self.orientation_arrows['z'].setRotation(np.rad2deg(orientation[2]))

    def _update_decode_arrow(self):
        decoded_direction = self._handle_direction_decode()
        self._handle_show_or_hide_arrow(decoded_direction)
        if self.logger is not None:
            self.logger.write(self.features, decoded_direction, int(self._prompt))

    def _update_feature_radial_plot(self):
        """
        Update the radial plot based on weighted feature magnitudes for each direction.
        """
        # Update the radial plot based on features and orientation
        orientation_offset = self.processor.wrist_orientation[0]  # Get device orientation (assume in radians)
        
        # Calculate the weighted sum of features for each direction
        # weighted_features = {
        #     'up': np.sum(self.adjusted_magnitude * weights['up']),
        #     'down': np.sum(self.adjusted_magnitude * weights['down']),
        #     'left': np.sum(self.adjusted_magnitude * weights['left']),
        #     'right': np.sum(self.adjusted_magnitude * weights['right'])
        # }
        weighted_features = {
            'up': np.sum(self.adjusted_magnitude[0]),
            'down': np.sum(self.adjusted_magnitude[4]),
            'left': np.sum(self.adjusted_magnitude[6]),
            'right': np.sum(self.adjusted_magnitude[2])
        }

        # print(weighted_features['up'])
        # Define the directions and corresponding angles for the radial plot
        directions = ['up', 'down', 'left', 'right']
        base_angles = [PI / 2, 3 * PI / 2, PI, 0]  # Adjust base angles for each direction (90, 270, 180, 0 degrees)
        
        # Update each feature curve based on the weighted values and orientation offset
        for i, direction in enumerate(directions):
            feature_value = weighted_features[direction]
            angle = base_angles[i] + orientation_offset
            
            x = [0, feature_value * np.cos(angle)]
            y = [0, feature_value * np.sin(angle)]
            
            # Update the radial curve for this direction
            self.feature_curves[i].setData(x, y)


    @staticmethod
    def get_colormap_colors(colormap_name: str, num_colors: int) -> list[tuple[int, int, int]]:
        """
        Generate a list of RGB colors from a specified Matplotlib colormap.

        Args:
            colormap_name (str): Name of the colormap (e.g., "coolwarm", "cool", "autumn").
            num_colors (int): Number of colors to sample.

        Returns:
            list[tuple[int, int, int]]: List of RGB colors as (R, G, B) tuples (0-255).
        """
        cmap = cm.get_cmap(colormap_name, num_colors)  # Load colormap with num_colors
        return [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_colors)]  # Convert to 0-255 RGB

    @staticmethod
    def generateBaseColors(palette_type: str = "default") -> list[tuple[int, int, int]]:
        """
        Generates a palette of RGB colors based on the specified palette type.

        Args:
            palette_type (str, optional): The type of color palette to generate. 
                - "default": Returns the default 8-color palette.
                - "warm": Returns a palette of 16 warm colors (reds, oranges, yellows).
                - "cool": Returns a palette of 8 cool colors (blues, purples, teals).

        Returns:
            list[tuple[int, int, int]]: A list of RGB tuples, where each tuple contains 
            three integers (0-255) representing a color.

        Example:
            >>> RealTimePlotter.generateBaseColors()
            [(48, 64, 166), (70, 118, 238), (27, 207, 213), ...]

            >>> RealTimePlotter.generateBaseColors("warm")
            [(255, 76, 51), (255, 127, 76), (229, 51, 25), ...]

            >>> RealTimePlotter.generateBaseColors("cool")
            [(51, 102, 255), (76, 127, 255), (51, 51, 229), ...]
        """
        match palette_type.lower():
            case "warm":
                # 16 warm colors (reds, oranges, yellows)
                warm_colors = RealTimePlotter.get_colormap_colors("autumn", 64)
                return warm_colors

            case "cool":
                # 8 cool colors (blues, purples, teals)
                cool_colors = RealTimePlotter.get_colormap_colors("cool", 8)
                return cool_colors

            case _:
                # Default 8 colors (same as before)
                default_colors = [
                    (0.1900, 0.2518, 0.6522),
                    (0.2769, 0.4658, 0.9370),
                    (0.1080, 0.8127, 0.8363),
                    (0.3857, 0.9896, 0.4202),
                    (0.8207, 0.9143, 0.2063),
                    (0.9967, 0.6082, 0.1778),
                    (0.8568, 0.2250, 0.1276),
                    (0.6896, 0.1158, 0.0806)
                ]
                return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in default_colors]