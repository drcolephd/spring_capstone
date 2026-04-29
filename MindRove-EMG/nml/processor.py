import numpy as np
from mindrove.board_shim import BoardShim
from numpy.typing import NDArray
from nml.binary_logger import BinaryLogger
from nml.binary_reader import BinaryReader
from nml.model_interactor import ModelInteractor
from nml.cluster_selection_window import ClusterSelectionWindow
from nml.spikes import BeamformerSpikeHandler
from nml.connections.parameter_socket import ParameterSocket
from nml.connections.stream_socket import StreamSocket
from PyQt5.QtGui import QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QTimer
from typing import Tuple
from nml.feature_weights import model
from nml.processor_modes import ProcessorMode
from nml.decoder import Decoder
from nml.processing import (
    EnvelopeSmoother, GyroProcessor, HighPassFilter,
    ImuArFeatureExtractor, ImuArModel,
    MlpWindowExtractor,
    MotorTwitchHandler, OrientationEstimator,
    SpatialReference, MontageMode,
    DEFAULT_ZCA_PARAMS, ZcaHandler, ZcaParams
)
from nml.gesture.mlp_model import MlpGestureModel

class Processor(QObject):
    board_shim: BoardShim | None = None
    spike = pyqtSignal(object, int, int)
    motor_units = pyqtSignal(object)
    cluster_color_assigned = pyqtSignal(int, QColor)
    auto_thresholds = pyqtSignal(object)
    omega = pyqtSignal(object, object, object, object, bool)
    delta_omega = pyqtSignal(float, float, float, float)
    embedding = pyqtSignal(float, float, int, int, float, int)

    emg_features = pyqtSignal(object)        # NDArray[float32] — combined EMG+IMU feature vector
    gesture_prediction = pyqtSignal(int, float)  # (class_label, confidence)

    parameter_socket: ParameterSocket | None = None
    stream_socket: StreamSocket | None = None
    _gesture_model: "MlpGestureModel | None" = None
    
    _decoder: Decoder | None = None
    _decode_mode: ProcessorMode = ProcessorMode.ANGULAR_VELOCITY
    @property
    def decode_mode(self) -> ProcessorMode:
        return self._decode_mode

    circular_buffer: "np.ndarray[np.float64]"  = None # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    _env_history: "np.ndarray[np.float64]" = None # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    _PLS_BETA: "np.ndarray[np.float32]" = model['coeff'] # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    
    
    _rates_buffer = []
    _filename: str | None = None
    _suffix: int | None = None

    orientation: "np.ndarray[np.float32]" = np.zeros(3, dtype=np.float32) # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    wrist_orientation: "np.ndarray[np.float32]" = np.zeros(3, dtype=np.float32) # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    _omega: "np.ndarray[np.float32]" = np.zeros(2, dtype=np.float32) # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    _delta_omega: "np.ndarray[np.float32]" = np.zeros(2, dtype=np.float32) # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    _omega_gain: "np.ndarray[np.float32]" = np.array([0.02, 0.02], dtype=np.float32) # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    _omega_threshold: "np.ndarray[np.float32]" = np.array([0.02, 0.015], dtype=np.float32) # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    # _omega_threshold: "np.ndarray[np.float32]" = np.array([0.015, 0.01], dtype=np.float32)
    cluster_selection_window: ClusterSelectionWindow | None = None
    filter_cutoff_spikes = 5.0 # Hz
    _montage = None
    _acc_channels: Tuple[int, int, int] = [20, 21, 22] # pyright: ignore[reportAssignmentType]
    _gyro_channels: Tuple[int, int, int] = [23, 24, 25] # pyright: ignore[reportAssignmentType]
    _gyro_channel_scalar: float | None = None
    _alpha: float = 0.98 # gain on gyro contribution to orientation; accelerometer contribution is 1-_alpha.
    _beta: float = 0.020
    _wrist_orientation_estimate_gain: float = 1.0
    _recording: bool = False
    _has_muaps: bool = False
    _logger: BinaryLogger | None = None
    _logger_file: str | None = None
    _batch = 0
    _sample = 0
    _env_buffer_size: int = 16384
    _has_rates_model: bool = False
    _collecting_rates_model: bool = False
    _rates_model: "np.ndarray[np.float32]" = None # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
    _dx: float = 0.0
    _x: float = 0.0
    _dy: float = 0.0
    _y: float = 0.0
    _xt: float = 0.0
    _yt: float = 0.0
    _button: bool = False
    _x_offset: float = 0.4
    _y_offset: float = 0.0
    _x_degrees: float = 20.0
    _y_degrees: float = 20.0
    _xrange: Tuple[float, float] = [2.515625, 3.34375] # pyright: ignore[reportAssignmentType]
    _yrange: Tuple[float, float] = [3.5625, 4.375] # pyright: ignore[reportAssignmentType]
    _omega_limit: float = 0.11*np.pi

    _sample_update_timer: QTimer | None = None
    _sample_timer_period = 5 # milliseconds

    _xy_interpolant_timer: QTimer | None = None
    _xy_interpolant_timer_period: int = 2 # milliseconds

    _hpf: HighPassFilter | None = None
    _spatial_ref: SpatialReference | None = None
    _zca: ZcaHandler | None = None

    def __init__(self, 
                 board_shim, 
                 buffer_size, 
                 num_channels: int=35, 
                 sample_rate=500, 
                 filter_cutoff_env: float=1.0, 
                 filter_order_env: int=1, 
                 filter_cutoff_hpf: float=100.0, 
                 filter_order_hpf: int=1, 
                 montage_mode: MontageMode=MontageMode.MONOPOLAR, 
                 channel: int=None, 
                 threshold: float=None, 
                 pre_peak_samples: int=6, 
                 post_peak_samples: int=9, 
                 num_ipts: int = 12, 
                 compute_thresholds: bool = True, 
                 do_spike_detection: bool = True, 
                 spike_handler_cls=BeamformerSpikeHandler, 
                 zca_params: ZcaParams = DEFAULT_ZCA_PARAMS):
        
        super(Processor, self).__init__()

        try:
            self._decode_mode = ProcessorMode(int(model["mode"]))
        except Exception:
            # Fallback: angular velocity mode
            self._decode_mode = ProcessorMode.ANGULAR_VELOCITY

        self.board_shim = board_shim
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.filter_order = filter_order_hpf
        self.spikes = spike_handler_cls(self, 
                 buffer_size = buffer_size, 
                 pre_peak_samples = pre_peak_samples,
                 post_peak_samples = post_peak_samples, 
                 enable = do_spike_detection, 
                 compute_thresholds = compute_thresholds, 
                 num_ipts = num_ipts, 
                 channel=channel, 
                 threshold=threshold)
        self.spikes.detected.connect(self.spike.emit)
        self.spikes.embedding.connect(self.embedding.emit)
        self.parameter_socket = ParameterSocket(num_channels=num_ipts, scale_factor=100.0)
        self.stream_socket = StreamSocket()

        # ---- High-pass filter (8 EMG channels) ----
        self._hpf = HighPassFilter(
            num_channels=8,
            fs=sample_rate,
            cutoff_hz=filter_cutoff_hpf,
            order=filter_order_hpf
        )

        # ---- Envelope smoothing (rectified → LPF) ----
        self._env_smoother = EnvelopeSmoother(
            num_channels=8,
            fs=self.sample_rate,
            cutoff_hz=filter_cutoff_env,   # Usually 1 Hz
            order=filter_order_env,
        )

        # ---- Spatial-Montage/Referencing ----
        self._spatial_ref = SpatialReference(montage_mode=montage_mode)

        # ---- ZCA Whitening ----
        self._zca = ZcaHandler(fs=sample_rate, params=zca_params)

        # ---- Gyroscope processing ----
        gyro_scalar = 0.00025 * (1/sample_rate) # Determined empirically
        self._gyro_processor = GyroProcessor(scalar=gyro_scalar) 

        # ---- Orientation estimator ----
        self._orientation = OrientationEstimator(
            alpha=self._alpha,
            wrist_gain=self._wrist_orientation_estimate_gain,
        )

        # ---- Motor Twitch Handler (vibration activity) ----
        self._twitch = MotorTwitchHandler(fs=self.sample_rate)

        # ---- Gesture feature extraction (EMG window + IMU AR model) ----
        self._emg_window = MlpWindowExtractor(num_channels=8)
        self._imu_ar = ImuArModel(
            order=8, channels=3,
            fit_window=sample_rate,           # 1-second fitting window
            refit_interval=sample_rate // 4,  # refit every 250 ms
        )
        self._imu_ar_feat = ImuArFeatureExtractor(
            channels=3,
            window_samples=sample_rate // 5,  # 200 ms window
        )

        # Initialize buffers
        self.circular_buffer = np.zeros((num_channels, buffer_size)).astype(np.float64)
        self._env_history = np.zeros((9, self._env_buffer_size)).astype(np.float64)

        self._rates_x = np.ones(num_ipts+1, dtype=np.float32)
        self._xy_interpolant_timer = QTimer()
        self._xy_interpolant_timer.timeout.connect(self._interpolate_xy)
        self._sample_update_timer = QTimer()
        self._sample_update_timer.timeout.connect(self.sample_device)
        # Decoder encapsulating decoding logic
        self._decoder = Decoder(self)

    def update(self, new_data):
        """
        Core update method: filtering → spatial → whitening → smoothing →
        IMU/orientation → twitch → spike detection → decoding → logging.
        """

        num_samples = new_data.shape[1]
        if num_samples < 1:
            return None

        # ---------------------------------------------------------
        # Update sample/batch indices
        # ---------------------------------------------------------
        self._sample = (self._sample + num_samples) % 32768
        self._batch = (self._batch + 1) % 32768
        sample = self._sample
        batch = self._batch

        # ---------------------------------------------------------
        # 1. High-pass filter (8 EMG channels)
        # ---------------------------------------------------------
        filtered_data = self._hpf.process(new_data[:8, :])

        # ---------------------------------------------------------
        # 2. Spatial referencing
        # ---------------------------------------------------------
        spatial_ref_data = self._spatial_ref.process(filtered_data)

        # ---------------------------------------------------------
        # 3. ZCA Whitening
        # ---------------------------------------------------------
        if not self._zca.trained:
            self._zca.update_buffer(spatial_ref_data)
            processed_data = spatial_ref_data
        else:
            processed_data = self._zca.apply(spatial_ref_data)

        # ---------------------------------------------------------
        # 4. Envelope smoothing (rectified → LPF)
        # ---------------------------------------------------------
        env_data = np.zeros((9, num_samples))
        env_data[1:9] = self._env_smoother.process(processed_data)

        # ---------------------------------------------------------
        # 5. IMU processing → gyro + accelerometer + orientation
        # ---------------------------------------------------------
        acc_samples = new_data[self._acc_channels, :]            # shape: (3, N)
        gyro_samples = new_data[self._gyro_channels, :]          # shape: (3, N)

        # (a) Gyroscope contribution (integrated via scalar gain)
        gyro_contrib = self._gyro_processor.process(gyro_samples)

        # (b) Accelerometer → gravity-subtracted tilt angles
        acc_mean = np.mean(acc_samples, axis=1)
        acc_angles = self._orientation.compute_acc_angles(acc_mean)

        # (c) Update orientation filter
        self._orientation.update(gyro_contrib, acc_angles)

        # Update public-facing orientation values
        self.orientation = self._orientation.orientation.copy()
        self.wrist_orientation = self._orientation.wrist_orientation.copy()

        # ---------------------------------------------------------
        # 5b. Gesture feature extraction (EMG window + IMU AR)
        # ---------------------------------------------------------
        for i in range(num_samples):
            emg_feat = self._emg_window.push_frame(processed_data[:, i])
            self._imu_ar.update(gyro_samples[:, i])
            self._imu_ar_feat.push(self._imu_ar.residual)
            if emg_feat is not None:
                combined = np.concatenate([emg_feat, self._imu_ar_feat.extract()])
                self.emg_features.emit(combined)
                if self._gesture_model is not None:
                    label, conf = self._gesture_model.predict(combined)
                    self.gesture_prediction.emit(label, conf)

        # ---------------------------------------------------------
        # 6. Twitch detection (vibration activity)
        # ---------------------------------------------------------
        twitch_value = self._twitch.process(acc_samples)

        # ---------------------------------------------------------
        # 7. Update circular buffer
        # ---------------------------------------------------------
        self.circular_buffer[:, :-num_samples] = self.circular_buffer[:, num_samples:]
        self.circular_buffer[:8, -num_samples:] = processed_data
        self.circular_buffer[8:, -num_samples:] = new_data[8:, :]

        # ---------------------------------------------------------
        # 8. Update long-term envelope history
        # ---------------------------------------------------------
        num_env_samples = min(num_samples, self.buffer_size)
        self._env_history[:, :-num_env_samples] = self._env_history[:, num_env_samples:]
        self._env_history[1:9, -num_env_samples:] = env_data[1:9, -num_env_samples:]

        # ---------------------------------------------------------
        # 9. Spike detection
        # ---------------------------------------------------------
        self.spikes.update(batch, sample, num_samples, processed_data)

        # ---------------------------------------------------------
        # 10. Decoder (movement decoding)
        # ---------------------------------------------------------
        if self._decoder is not None and self._decode_mode is not ProcessorMode.OFF:
            self._decoder.step(env_data)

        # ---------------------------------------------------------
        # 11. Binary logging (if active)
        # ---------------------------------------------------------
        if self._recording:
            self._logger.write_batch(
                processed_data.T,
                batch,
                sample,
                self._x, self._y,
                float(self.orientation[0]),
                self._button
            )



    def get_covariates(self) -> Tuple[float, float, float, bool]:
        return (self._x, self._y, float(self.orientation[0]), self._button)

    def start_device_sampling(self):
        self._sample_update_timer.start(self._sample_timer_period)

    def stop_device_sampling(self):
        self._sample_update_timer.stop()

    def start_xy_interpolation(self):
        self._xy_interpolant_timer.start(self._xy_interpolant_timer_period)

    def stop_xy_interpolation(self):
        self._xy_interpolant_timer.stop()

    def sample_device(self):
        """Get new data from the device, into our software buffer."""
        board_data = self.board_shim.get_board_data()
        if board_data is not None:
            self.update(board_data)

    def _interpolate_xy(self):
        x = 0.75 * self._x + 0.25 * self._xt
        y = 0.75 * self._y + 0.25 * self._yt
        self._dx = self._x - x
        self._x = x
        self._dy = self._y - y
        self._y = y

    @pyqtSlot(int)
    def set_montage(self, mode: int = 2):
        """Sets the spatial reference montage for EMG channels.
        
        0: Monopolar
        1: Single-Differential
        2: Discrete Spatial Laplacian (default)
        """
        self._montage_mode = mode
        if mode == 2:
            self._montage = [
                    [1, 7],  # Channel 0 has neighbors 1 and 7
                    [0, 2],  # Channel 1 has neighbors 0 and 2
                    [1, 3],  # Channel 2 has neighbors 1 and 3
                    [2, 4],  # Channel 3 has neighbors 2 and 4
                    [3, 5],  # Channel 4 has neighbors 3 and 5
                    [4, 6],  # Channel 5 has neighbors 4 and 6
                    [5, 7],  # Channel 6 has neighbors 5 and 7
                    [6, 0]   # Channel 7 has neighbors 6 and 0
                ]
        elif mode == 1:
            self._montage = [[1], [2], [3], [4], [5], [6], [7], [0]]
        else:
            self._montage = [[], [], [], [], [], [], [], []]

    def update_position(self, x_in: float, y_in: float, button_state: bool):
        self._xt = Processor.remap(x_in, self._xrange, self._x_offset, self._x_degrees)
        self._yt = Processor.remap(y_in, self._yrange, self._y_offset, self._y_degrees)
        self._button = button_state

    def update_rates_buffer(self, rates):
        """
        Stores historical movement and coefficient data for directional trend analysis.
        """
        self._rates_x[1:] = rates
        self._rates_buffer.append({
            "x": self._dx, 
            "y": self._dy, 
            "sample": rates
        })

    @pyqtSlot(object)
    def on_model_update(self, new_coefficients: "np.ndarray[np.float32]"): # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
        self._PLS_BETA = new_coefficients

    def start_emg_only_recording(self, fname_emg: str, fname_rates: str = None):
        self._recording = True
        self._logger = BinaryLogger(fname_emg)
        self._logger_file = fname_emg
        if fname_rates is not None and self.spikes.enabled:
            self.spikes.start_recording(fname_rates)

    def stop_emg_only_recording(self):
        self._recording = False
        self._logger.close()
        del self._logger
        self._logger = None
        r = BinaryReader(self._logger_file)
        r.convert()
        r.close()
        del r
        self._logger_file = None
        if self.spikes.enabled:
            self.spikes.stop_recording()

    @pyqtSlot(float)
    def update_omega_limit(self, base_gain: float):
        self._omega_limit = 0.11*np.pi / base_gain

    @pyqtSlot(object)
    def update_omega_deadzone(self, new_threshold: "np.ndarray[np.float32]"): # pyright: ignore[reportInvalidTypeArguments, reportAssignmentType]
        self._omega_threshold = new_threshold

    def get(self) -> NDArray[np.float64]:
        return self.circular_buffer # pyright: ignore[reportReturnType]
    
    def set_mode(self, mode: int | ProcessorMode):
        """
        Set the decode mode.

        Accepts either:
        - an int in [0, 4], for backward compatibility, or
        - a ProcessorMode enum.
        """
        if isinstance(mode, ProcessorMode):
            self._decode_mode = mode
            return

        # Backward-compatible int handling
        if (mode < 0) or (mode > 4):
            raise Exception("mode must be integer from 0 to 4!")
        self._decode_mode = ProcessorMode(mode)
    
    def set_orientation(self, new_orientation = None):
        """Sets the zero values for orientation."""
        if new_orientation is None:
            self.orientation = np.zeros(3)
        else:
            self.orientation -= new_orientation

    def set_alpha(self, new_alpha):
        """Set new value for alpha and beta coefficients."""
        self._alpha = new_alpha
        self._beta = 1 - new_alpha
    
    @pyqtSlot(bool)
    def handle_rates_model_checkbox_click(self, checked: bool):
        if checked:
            self._collecting_rates_model = True
        else:
            self._collecting_rates_model = False
            if len(self._rates_buffer) > 100:
                tmp_name = self._filename.replace('data/', '').replace('\\','/')
                def_name = f"{tmp_name}_{self._suffix}_rates_model"
                mdl, mdl_path = ModelInteractor.perform_pls_regression(self._rates_buffer, n=4, def_name=def_name) # pyright: ignore[reportGeneralTypeIssues]
                saved_rates = mdl_path is not None
                if saved_rates:
                    self._rates_model = mdl
                    muap_filters_fname = mdl_path.replace("_rates_model", "_muap_filters")
                    self.spikes.save_muap_filters_and_thresholds(muap_filters_fname)
                else:
                    self._rates_model = None
                self._rates_buffer = []
                self._has_rates_model = saved_rates
            else:
                self._rates_buffer = []
                self._has_rates_model = False
                print("Insufficient samples. Rates model cleared.")

    def set_filename(self, filename: str, suffix: int):
        self._filename = filename
        self._suffix = suffix

    @staticmethod
    def remap(data: float, data_lims: Tuple[float, float], output_offset: float = 0.0, output_gain: float = 20.0) -> float:
        """
        Returns data remapped between -1.0 and 1.0 based on data limits.
        """
        data_c = (data_lims[0] + data_lims[1])/2
        data_r = (data_lims[1] - data_lims[0])/2
        return ((data - data_c) / data_r + output_offset) * output_gain

    @pyqtSlot(object)
    def set_gesture_model(self, model: "MlpGestureModel | None") -> None:
        """Attach (or clear) the gesture classifier used during live inference."""
        self._gesture_model = model

    def reset_gesture_extractors(self) -> None:
        """Reset the EMG window and IMU AR buffers (e.g. after a mode change)."""
        self._emg_window.reset()
        self._imu_ar.reset()
        self._imu_ar_feat.reset()

    def __del__(self):
        try:
            self._sample_update_timer.stop()
        except Exception:
            pass

        try:
            self._xy_interpolant_timer.stop()
        except Exception:
            pass

        try:
            if self.board_shim.is_prepared():
                self.board_shim.release_session()  # Clean up the board session
        except Exception:
            print("[Processor]::Board session already released.")