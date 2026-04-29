from PyQt5.QtCore import QObject, QTimer, pyqtSignal
import serial
import struct
import time
import numpy as np
from typing import Optional, Tuple
from serial.tools import list_ports
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi # sosfilt, sosfilt_zi

class Teensy(QObject):
    newData = pyqtSignal(object, bool)  # Signal to emit new data
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    sampling = pyqtSignal()
    stopped = pyqtSignal()

    TEENSY_VID = 0x16C0  # Vendor ID for Teensy
    TEENSY_PID = 0x0483  # Product ID for Teensy
    serial = None
    bits_range: np.float32 = 128.0
    bits_center: np.float32 = 0.0
    volts_center: np.float32 = 2.5
    volts_range: np.float32 = 2.0
    _connected: bool = False
    _sampling: bool = False
    _voltage_extreme: np.float32 = 5.0 # The "maximum voltage excursion" positive or negative for output
    _buffer_size: int = 2048          # Number of samples in past filtered data buffer
    _buffer: "np.ndarray[np.float32]"       = None # Past filtered data buffer.
    _filter_state: "np.ndarray[np.float32]" = None # Filter states for IIR LPF
    _filter_state_notch: "np.ndarray[np.float32]" = None # Filter states for IIR Notch filter
    _data: "np.ndarray[np.uint16]" = None    # "most-recent" data.
    _min: "np.ndarray[np.float32]" = None    # Minimum values from all samples.
    _max: "np.ndarray[np.float32]" = None    # Maximum values from all samples.
    # _gain: "np.ndarray[np.float32]" = None   # Gains for each channel
    _emission_counter: int = 0      # tracks poll iterations
    _emission_threshold: int = 15   # poll iterations before emitting `newData` signal
    _poll_interval: int = 2         # milliseconds
    _filter_cutoff: float =  1.5    # Hz, LPF cutoff frequency
    _filter_order: int = 7          # LPF Butterworth Filter Order
    _notch_frequency: float = 60.0  # Notch center frequency
    _b_lp = _a_lp = None            # Transfer function coefficients for IIR LPF Butterworth filter
    _b_notch = _a_notch = None      # Transfer function coefficients for IIR Notch filter

    # Kalman parameters
    _prev_sample: "np.ndarray[np.float32]" = None
    kalman_state: "np.ndarray[np.float32]" = None
    kalman_covariance: Tuple["np.ndarray[np.float32]", "np.ndarray[np.float32]"] = None 
    kalman_process_noise: Tuple["np.ndarray[np.float32]", "np.ndarray[np.float32]"] = None # process noise variance
    kalman_measurement_noise: Tuple["np.ndarray[np.float32]", "np.ndarray[np.float32]"] = None # measurement noise variance
    kalman_buffer: Tuple["np.ndarray[np.float32]", "np.ndarray[np.float32]"] = None
    kalman_buffer_index: "np.ndarray[int]" = None
    kalman_samples: int = None
    _kalman_buffer_full: bool = False
    _kalman_dt: float = None
    _kalman_delta_threshold: "np.ndarray[np.float32]" = None

    def __init__(self, port: Optional[str] = None, baud_rate: int = 115200, timeout: int = 1, poll_interval: int = 2, emission_threshold: int = 15, n_channels: int = 2, filter_cutoff: float = 5.0, filter_order: int = 1, buffer_size: int = 2048, notch_frequency: float = 60.0, parent=None):
        """
        Initialize the Teensy interface and set up polling.

        Parameters:
        - port (str): Optional; the serial port name (e.g., 'COM3' on Windows or '/dev/ttyACM0' on Linux).
        - baud_rate (int): The baud rate for serial communication, matching the Teensy rate; default 115200.
        - timeout (float): The read timeout for the serial connection in seconds; default is 1.
        - poll_interval (int): Interval in milliseconds for polling data; default is 2.
        - emission_threshold (int): Number of times to poll before emitting a `newData` signal; default is 15.
        - n_channels (int): Number of analog channels to read from; default is 2.
        - filter_cutoff (float): Lowpass cutoff frequency (Hz); default is 5.0.
        - filter_order (int): Butterworth Lowpass Filter order; default is 1.
        - notch_frequency (float): Notch filter center frequency; default is 60.0 (Hz).
        - buffer_size (int): The size of the buffer for applying the IIR lowpass filter; default is 2048.
        """
        super().__init__(parent)
        self.port = port or self._find_teensy_port()
        if not self.port:
            raise ConnectionError("Teensy device not found. Ensure it is connected and try again.")
        
        self._filter_cutoff = filter_cutoff
        self._filter_order = filter_order
        self._notch_frequency = notch_frequency
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.n_channels = n_channels

        # Timer setup for periodic polling
        self._polling_timer = QTimer()
        self._polling_timer.timeout.connect(self._poll)
        self._poll_interval = poll_interval
        self._emission_counter = 0
        self._emission_threshold = emission_threshold
        self._sampling = False  # Sampling flag
        self._buffer_size = buffer_size

        # Most recent sample data storage
        self._buffer = np.zeros((n_channels, self._buffer_size), dtype=np.float32)
        self._data = np.zeros(n_channels, dtype=np.uint16)
        self._min = np.full(n_channels, self._voltage_extreme, dtype=np.float32)
        self._max = np.full(n_channels, -self._voltage_extreme, dtype=np.float32)
        # self._gain = np.full(n_channels, -1.0, dtype=np.float32)
        self._initialize_filter()
        self._initialize_kalman()

        # Allow time for the connection to establish
        time.sleep(2)
        print(f"Connected to Teensy on port {self.port}")

    def connect(self):
        self.serial = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
        self._connected = True
        self.connected.emit()

    def disconnect(self):
        if self._connected:
            if self._sampling:
                self.stop()
            del self.serial
            self.serial = None
            self.disconnected.emit()

    def _find_teensy_port(self) -> Optional[str]:
        """
        Searches for the Teensy device by its VID and PID.

        Returns:
        - str: The port name if a Teensy device is found, otherwise None.
        """
        ports = list_ports.comports()
        for port in ports:
            if port.vid == self.TEENSY_VID and port.pid == self.TEENSY_PID:
                return port.device
        return None

    def start(self):
        """Starts the timer for polling the Teensy at the specified interval."""
        if not self._sampling and self._connected:
            self._sampling = True
            self._polling_timer.start(self._poll_interval)
            self.sampling.emit()
        

    def stop(self):
        """Stops the timer and polling of the Teensy device."""
        if self._sampling:
            self._polling_timer.stop()
            self._sampling = False
            self.stopped.emit()

    def _initialize_filter(self): 
        nyquist_rate = 500 / self._poll_interval  # _poll_interval in milliseconds
        cutoff_norm = self._filter_cutoff / nyquist_rate
        self._b_lp, self._a_lp = butter(self._filter_order, cutoff_norm, btype='low')
        self._filter_state_lp = np.array([lfilter_zi(self._b_lp, self._a_lp) * 0 for _ in range(self.n_channels)], dtype=np.float32)
    
        # Notch filter setup for 60 Hz
        quality_factor = 30.0  # Quality factor for the notch filter
        notch_norm = self._notch_frequency / nyquist_rate
        self._b_notch, self._a_notch = iirnotch(notch_norm, quality_factor)
        self._filter_state_notch = np.array([lfilter_zi(self._b_notch, self._a_notch) for _ in range(self.n_channels)], dtype=np.float32)


    def scale_data(self) -> np.ndarray:
        """
        Scales the raw integer data in `self._data` to voltage values based on the configured
        bits_center, bits_range, volts_center, and volts_range attributes.

        Returns:
        - np.ndarray: The scaled data as a float32 array, representing voltage values.
        """
        # Normalize the data around bits_center and scale it within [-1, 1]
        normalized_data = (self._data.astype(np.float32) - self.bits_center) / self.bits_range
        
        # Scale the normalized data to the voltage range and center it at volts_center
        scaled_data = (normalized_data * self.volts_range) + self.volts_center
        
        return scaled_data

    def _poll(self):
        """Polls the Teensy device for new readings and updates _data."""
        try:
            # Send the number of channels as a single byte
            self.serial.write(bytes([self.n_channels]))
            
            # Calculate expected number of bytes and read response
            expected_bytes = self.n_channels + 1  # 2 bytes per channel + 1 button byte
            data = self.serial.read(expected_bytes)
            readings = struct.unpack(f'>{self.n_channels}bB', data)
            # print(f"{readings[0]},{readings[1]}")
            self._data = np.array(readings[:self.n_channels], dtype=np.int16)  # Extract only the analog readings
            button_state = bool(readings[-1] == 1)  # The last byte is the button state (1 = pressed, 0 = not pressed)
            
            # Scale data to voltage range
            scaled_data = self.scale_data()
            # print(f"{scaled_data[0]},{scaled_data[1]}")
            # Update buffer with the latest sample
            self._buffer = np.roll(self._buffer, -1, axis=1)
            self._buffer[:, -1] = scaled_data

            # Apply lowpass and notch filters on the latest sample in each channel
            filtered_sample = np.zeros(self.n_channels, dtype=np.float32)
            for i in range(self.n_channels):
                # Notch filter
                filtered_notch, self._filter_state_notch[i] = lfilter(
                    self._b_notch, self._a_notch, [self._buffer[i, -1]], zi=self._filter_state_notch[i]
                )
                # Lowpass filter
                filtered_lp, self._filter_state_lp[i] = lfilter(
                    self._b_lp, self._a_lp, filtered_notch, zi=self._filter_state_lp[i]
                )
                
                filtered_sample[i] = filtered_lp

            # kalman_filtered_sample = self._apply_kalman_filter(filtered_sample)
            # for i in range(self.n_channels):
            #     self._prev_sample[i] = np.mean(self.kalman_buffer[i][:,0])

            # Update min and max values for each channel
            self._min = np.minimum(self._min, filtered_sample)
            self._max = np.maximum(self._max, filtered_sample)

            # Emit new data if the emission threshold is reached
            self._emission_counter = (self._emission_counter + 1) % self._emission_threshold
            if self._emission_counter == 0:
                self.newData.emit(filtered_sample, button_state)  # Emit new data as a signal
        except serial.SerialException as e:
            print(f"Error reading from Teensy: {e}")
            self.stop()

    def reset_limits(self):
        """
        Resets the min and max value history for all channels.
        """
        self._min = np.full(self.n_channels, self._voltage_extreme, dtype=np.float32)
        self._max = np.full(self.n_channels, -self._voltage_extreme, dtype=np.float32)

    def get_limits(self, x_channel: int = 0, y_channel: int = 1) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns the _min and _max values for x and y channels.
        """
        v_max = float(self._voltage_extreme)
        v_min = -1.0 * float(self._voltage_extreme)
        x_range = [max(min(float(self._min[x_channel]), v_max), v_min), max(min(float(self._max[x_channel]), v_max), v_min)]
        y_range = [max(min(float(self._min[y_channel]), v_max), v_min), max(min(float(self._max[y_channel]), v_max), v_min)]
        return [x_range, y_range]

    def sample(self, n_channels: Optional[int] = None) -> Tuple[int, ...]:
        """
        Requests a specified number of analog readings from the Teensy device.

        Parameters:
        - n_channels (int): Number of analog channels to expect data from (2 to 8). If None, uses self.n_channels.

        Returns:
        - tuple: (reading0, reading1, ..., readingN) where each reading is an integer 
        (0-4095, representing 12-bit resolution).

        Raises:
        - RuntimeError: If the response is not in the expected format.
        """
        n_channels = n_channels or self.n_channels
        if not (2 <= n_channels <= 8):
            raise ValueError("n_channels must be between 2 and 8.")

        # Send request to Teensy
        self.serial.write(bytes([n_channels]))

        # Read expected bytes
        expected_bytes = n_channels * 2 + 1
        data = self.serial.read(expected_bytes)

        if len(data) != expected_bytes:
            raise RuntimeError("Incomplete data received from Teensy.")

        return struct.unpack(f'>{n_channels}HB', data)

    def getData(self) -> np.ndarray:
        """Returns the most recent data sample."""
        return self._data

    def close(self):
        """Closes the serial connection and stops sampling if active."""
        self.stop()
        if self.serial.is_open:
            self.serial.close()

    def _initialize_kalman(self):
        """Initializes Kalman filter parameters for each channel."""
        self._prev_sample = np.zeros(self.n_channels, dtype=np.float32)
        self.kalman_samples = 10
        self.kalman_state = np.zeros((self.n_channels, 2), dtype=np.float32) # Each channel has (position, velocity)
        self.kalman_covariance = [np.eye(2, dtype=np.float32) for _ in range(self.n_channels)]  # Initial covariance
        self.kalman_process_noise = [np.array([[0.001, 0.00],[0.00, 0.001]],dtype=np.float32) for _ in range(self.n_channels)] # Process noise variance
        self.kalman_measurement_noise = [np.array([[0.08, 0.000],[0.000, 0.08]],dtype=np.float32) for _ in range(self.n_channels)]  # Measurement noise variance
        self.kalman_buffer = [np.zeros((self.kalman_samples, 2), dtype=np.float32) for _ in range(self.n_channels)]
        self.kalman_buffer_index = np.zeros(self.n_channels, dtype=int)
        self._kalman_delta_threshold = np.array([0.04, 0.04], dtype=np.float32)
        self._kalman_buffer_full = False
        self._kalman_dt = self._poll_interval / 1000.0 # Convert milliseconds to seconds for time step

    def _apply_kalman_filter(self, measurements) -> "np.ndarray[np.float32]":
        filtered = np.zeros(self.n_channels, dtype=np.float32)
        
        for i in range(self.n_channels):
            # Update rolling buffer with the new measurement and estimated delta
            delta = self._prev_sample[i] - measurements[i]
            if delta < self._kalman_delta_threshold[i]:
                delta = 0
            obs_state = np.array([measurements[i], delta / self._kalman_dt], dtype=np.float32)
            self.kalman_buffer[i][self.kalman_buffer_index[i] % self.kalman_samples,:] = obs_state
            self.kalman_buffer_index[i] = (self.kalman_buffer_index[i] + 1) % self.kalman_samples

            if not self._kalman_buffer_full:
                self._kalman_buffer_full = self.kalman_buffer_index[i] == 0
            
            # Estimate process and measurement noise from recent sample covariance
            if self._kalman_buffer_full:
                history_covariance = np.cov(self.kalman_buffer[i], rowvar=False)
                measurement_noise = self.kalman_measurement_noise[i] * np.linalg.norm(history_covariance)
            else:
                # Default noise when history is insufficient
                measurement_noise = self.kalman_measurement_noise[i]

            # Prediction step: predict the next position and velocity
            F = np.array([[1, self._kalman_dt], [0, 1]])  # State transition model
            pred_state = F @ self.kalman_state[i]
            pred_covariance = F @ self.kalman_covariance[i] @ F.T + self.kalman_process_noise[i]

            # Measurement update step
            H = np.eye(2)  # Observation model
            kalman_gain = pred_covariance @ np.linalg.inv(pred_covariance + measurement_noise)
            self.kalman_state[i] = pred_state + kalman_gain @ (obs_state - H @ pred_state)
            self.kalman_covariance[i] = (np.eye(2) - kalman_gain) @ pred_covariance

            # Store the filtered position for output (first element of the state vector)
            filtered[i] = self.kalman_state[i][0]  # Position state
        return filtered


    def __del__(self):
        """Ensure connection is closed on deletion."""
        if self.serial is not None:
            if self.serial.is_open:
                self.serial.close()

    def __enter__(self):
        """Context manager entry; returns self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit; ensures the serial connection is closed."""
        self.close()
