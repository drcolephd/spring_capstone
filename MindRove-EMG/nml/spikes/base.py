from abc import abstractmethod
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
from scipy.signal import butter, sosfilt, sosfilt_zi
import numpy as np
from nml.spike_logger import SpikeLogger
from nml.spike_reader import SpikeReader
from typing import Optional
from collections import Counter
import pickle, os

class BaseSpikeHandler(QObject):
    detected = pyqtSignal(object, int, int)  # (waveform, channel, sample index)
    embedding = pyqtSignal(float, float, int, int, float, int)  # (x, y, sample, ch, orientation, id)
    threshold = pyqtSignal(object)
    source_thresholds_changed = pyqtSignal(object)
    num_ipts_changed = pyqtSignal(int)
    enabled: bool = True

    _verbose: bool = False
    _model_file: str = "max_muaps_model.pkl"
    _precision_best: float = 0.01
    _precision_worst: float = 0.2
    _precision: float = 0.1

    _processor = None
    _batch: int = 0
    _sample: int = 0
    _source: int = 0
    _scalar: float = 0.5
    _spread = None
    _history_orientation: "np.ndarray[np.float64]" = np.zeros(3, dtype=np.float64)
    _history_acc: float = 0.0
    
    _sos_spikes = None
    _filter_state_spikes = None

    _has_muaps: bool = False
    _needs_initial_clustering: bool = False

    _collecting_spike_buffer: bool = False
    _collecting_covariance_buffer: bool = True
    
    _compute_threholds: bool = False
    _has_initial_history: bool = False

    _emg_history_buffer_size: int = 16384
    _emg_history: np.ndarray = None

    _enable_state_adaptation: bool = False
    _spike_buffer_index: int = 0
    _spike_buffer_size: int = None
    _spike_buffer_full: bool = False
    _spike_buffer: np.ndarray = None
    _current_buffer: "np.ndarray[np.float64]" = None
    _current_envelope: "np.ndarray[np.float64]" = None
    _ipt_buffer: "np.ndarray[np.float64]" = None
    _ipt_buffer_size: int = 256
    _ipt_normalized_buffer: "np.ndarray[np.float64]" = None
    _spike_channel: int = 0
    _spike_threshold: "np.ndarray[np.float32]" = np.full(8, 1500.0, dtype=np.float32)
    _pre_peak_samples: int = None
    _post_peak_samples: int = None

    _recording: bool = False
    _logger: SpikeLogger = None
    _logger_file: str = None

    __spike_metadata: list = []
    _state_orientation = None
    _state_activity = None

    max_amp = None
    _conditioning_noise_bw: float = 0.001
    _max_states: int = None
    _state: int = None
    _state_likelihoods: "np.ndarray[np.float64]" = None
    _state_likelihood_alpha: float = 0.05 # Smoothing alpha for state likelihood updates. Larger values allow state transitions more rapidly.
    _lb_log_likelihood: float = -10 # Clip so we can do a smoothed likelihood estimation
    
    _P = None  # Pseudo-inverse of power-whitening covariance matrix
    _R = None
    _muap_filters = None
    
    _ipt_max_values = None
    _threshold = None
    _limit = None

    _num_ipts: int = None
    _ipt_threshold = None
    _rates = None  # Stores spike rates
    _extension_factor: int = None # This is determined as pre_peak_samples + post_peak_samples + 1

    def __init__(self,
                 processor = None,  
                 buffer_size: int = 1024, 
                 pre_peak_samples: int = 14,
                 post_peak_samples: int = 17, 
                 enable: bool = True, 
                 compute_thresholds: bool = True, 
                 num_ipts: int = 64, 
                 channel: int = None, 
                 threshold: float = None, 
                 max_states: int = 10, 
                 model_file: str = "max_muaps_model.pkl"):
        super().__init__()
        self._processor = processor
        self._sample_rate = processor.sample_rate
        
        if threshold is not None:
            self._spike_threshold = np.full(8, threshold)
        if channel is not None:
            self._spike_channel = channel
        self._spike_buffer_size = buffer_size
        self._pre_peak_samples = pre_peak_samples
        self._post_peak_samples = post_peak_samples
        self._extension_factor = pre_peak_samples + post_peak_samples + 1
        self.enabled = enable
        self._compute_thresholds = compute_thresholds
        self._emg_history = np.zeros((
            8, self._emg_history_buffer_size
        )).astype(np.float64)
        self._spike_buffer = np.zeros((
            self._spike_buffer_size, 8, 
            self._extension_factor
        )).astype(np.float64)
        self._current_buffer = np.zeros((
            8, self._extension_factor
        )).astype(np.float64)
        self._current_envelope = np.zeros((
            8, self._extension_factor
        )).astype(np.float64)
        self.set_max_states(max_states)
        self.set_num_ipts(num_ipts)
        if model_file is not None:
            self._model_file = model_file
            self.load()

    def set_max_states(self, n: int):
        if self._state is None:
            self._state = 0
        else:
            self._state = min(self._state, self._max_states-1)
        self._max_states = n
        self.max_amp = [np.ones((8, 1),dtype=np.float32)] * self._max_states
        self._muap_filters = [None] * self._max_states
        self._P = [None] * self._max_states
        self._R = [None] * self._max_states
        self._spread = np.zeros(self._max_states)
        self._state_likelihoods = np.full(self._max_states, self._lb_log_likelihood)
        self._state_orientation = [None] * self._max_states
        self._state_activity = [None] * self._max_states

    def update(self, cur_batch: int, cur_sample: int, num_samples: int, data):
        """Core update loop for filtering/online update."""
        if not self.enabled:
            return
        self._batch = cur_batch
        self._sample = cur_sample
        k = min(num_samples, self._extension_factor)
        if self._collecting_covariance_buffer:
            self._emg_history[:, :-k] = self._emg_history[:, k:]
            self._emg_history[:, -k:] = data[:, -k:]
        
        self._current_buffer[:, :-k] = self._current_buffer[:, k:]
        self._current_buffer[:, -k:] = data[:, -k:]

        self._current_envelope[:, :-k] = self._current_envelope[:, k:]
        self._current_envelope[:, -k:] = self._processor._env_history[1:9, -k:]

        # Update orientation on very slow timescale. 
        beta = 1 - self._state_likelihood_alpha
        self._history_orientation = self._state_likelihood_alpha * self._processor.orientation + beta * self._history_orientation
        activity = BaseSpikeHandler.compute_activity_sma(self._processor.circular_buffer[self._processor._acc_channels, -1:])
        self._history_acc = self._state_likelihood_alpha * activity + beta * activity

        if (not self._has_initial_history) and (self._sample > self._emg_history_buffer_size) and (self._compute_thresholds):
            self._has_initial_history = True
            self.compute_auto_thresholds()
        self.handle_spike_detection(num_samples)
        self.handle_clustering()
        self.handle_recording()

    @staticmethod
    def compute_activity_sma(acc_data):
        """
        Computes Signal Magnitude Area (SMA) activity index.
        
        Args:
            acc_data (np.ndarray): (3, N) array where rows are [ax, ay, az] and columns are time samples.

        Returns:
            float: SMA activity index.
        """
        return np.abs(acc_data).sum()  # Sum across axes

    @pyqtSlot(object)
    def on_recruitment_order_change(self, order: "np.ndarray"):
        if self._muap_filters[self._state] is not None:
            self._muap_filters[self._state] = self._muap_filters[self._state][order[::-1], :]
            self._ipt_buffer = self._ipt_buffer[order[::-1],:]
            self._ipt_normalized_buffer = self._ipt_normalized_buffer[order[::-1],:]
            self._ipt_max_values[self._state] = self._ipt_max_values[self._state][order[::-1],:]
            # source_thresholds = np.mean(np.abs(self._ipt_buffer),axis=1)
            print("Updated recruitment order.")
            # self.source_thresholds_changed.emit(source_thresholds.flatten())

    def handle_recording(self):
        if self._recording:
            x, y, orientation, btn = self._processor.get_covariates()
            # print(f"self._num_ipts: {self._num_ipts} vs self._rates.size: {self._rates.size}")
            self._logger.write(self._rates, self._num_ipts, x, y, orientation, btn)

    @abstractmethod
    def handle_clustering(self):
        """
        Handles the actual separation of waveforms (from subclass of BaseSpikeHandler).
        """
        raise(NotImplementedError("handle_clustering is not implemented in BaseSpikeHandler. It should be overloaded in a subclass instead."))

    def handle_spike_detection(self, num_samples: int):
        # At the end of the update, adjust all spike IDs by subtracting `num_samples`
        self.__spike_metadata = [(spike_id - num_samples, ch) for (spike_id, ch) in self.__spike_metadata]
        # Remove expired spikes (IDs that have gone negative)
        self.__spike_metadata = [(spike_id, ch) for (spike_id, ch) in self.__spike_metadata if spike_id >= 0]
        self._detect_channel_spike(self._spike_channel)
        # Detect spikes on specified channel
        other_ch = [i for i in range(8) if i != self._spike_channel]
        for ch in other_ch:
            self._detect_channel_spike(ch)

    def compute_auto_thresholds(self, n_sd: float = 0.5):
        if self._has_initial_history:
            buffered_data = self._emg_history
        else:
            buffered_data = self._processor.circular_buffer[:8, :]
        # Calculate the mean and standard deviation across the history for each channel
        means = np.mean(buffered_data, axis=1)  # Mean for each channel
        stds = np.std(buffered_data, axis=1)    # Standard deviation for each channel
        for i in range(8):
            if means[i] < 0:
                self._spike_threshold[i] = means[i] - n_sd * stds[i]
            else:
                self._spike_threshold[i] = means[i] + n_sd * stds[i]
        new_thresholds = self._spike_threshold.copy()
        for (k,threshold) in enumerate(new_thresholds):
            print(f"EMG-Auto-Threshold: Channel-{k:d}: {threshold:.2f}")
        self.threshold.emit(new_thresholds)

    def set_detector(self, channel, threshold):
        """
        Set the channel and threshold for spike detection.
        """
        self._spike_channel = channel
        self._spike_threshold[self._spike_channel] = threshold

    @abstractmethod
    def save_muap_filters_and_thresholds(self, file_path: Optional[str] = None) -> None:
        """
        Save _muap_filters and _spike_threshold to separate CSV files.
        The threshold file name is derived from the MUAP filter file name.

        Parameters:
        - file_path (Optional[str]): Base path to save the CSV files. If None, a file dialog is used.
        """
        raise(NotImplementedError("save_muap_filters_and_thresholds is not implemented in BaseSpikeHandler. It should be overloaded in a subclass instead."))

    @abstractmethod
    def load_models(self, rates_model_path: str = None) -> None:
        """
        Load the _rates_model, _muap_filters, and _spike_thresholds attributes from
        CSV files with the same base name but different suffixes.

        Parameters:
        - rates_model_path (Optional[str]): Base filename (ends with _rates_model.csv) for loading. 
        If None, a file dialog is used to select the file.
        """
        raise(NotImplementedError("load_models is not implemented in BaseSpikeHandler. It should be overloaded in a subclass instead."))

    @abstractmethod
    def cluster(self, new_data: np.ndarray):
        raise(NotImplementedError("cluster method is not implemented in BaseSpikeHandler. It should be overloaded in a subclass instead."))

    @abstractmethod
    def _muaps(self)->"np.ndarray[np.float64]":
        raise(NotImplementedError("_muaps method is not implemented in BaseSpikeHandler. It should be overloaded in a subclass instead."))

    @pyqtSlot(int, object)
    def on_spike_scope_rates(self, n: int, new_rates: "np.ndarray[np.float32]"):
        """
        Slot which is connected to the SpikeScope to handle incoming rate estimation data updates.
        """
        # rates = np.zeros(n, dtype=np.float32)
        # for i in range(n):
            # rates[i], self._filter_state_spikes[i] = sosfilt(self._sos_spikes, np.array([new_rates[i]]), zi=self._filter_state_spikes[i])
        self._rates = new_rates
        self._processor.update_rates_buffer(new_rates)

    def _initialize_filter(self):
        nyquist = 0.5 * self._processor.sample_rate
        cutoff_spikes = self._processor.filter_cutoff_spikes / nyquist
        self._sos_spikes = butter(self._processor.filter_order, cutoff_spikes, btype='low', output='sos')
        self._filter_state_spikes = np.array([sosfilt_zi(self._sos_spikes) for _ in range(self._num_ipts)])

    def _detect_channel_spike(self, ch):
        """
        Detects spikes on an individual channel using threshold-crossings.
        """
        channel_data = self._processor.circular_buffer[ch, :]
        peak_indices = self._detect_threshold_crossings(channel_data, self._spike_threshold[ch])
        for peak_index in peak_indices:
            # Check if spike is new by comparing with existing metadata
            if not any(spike_id == peak_index for spike_id, _ in self.__spike_metadata):
                # This is a new spike, so buffer and emit it
                waveform = self.extract(channel_data, peak_index)
                self.detected.emit(np.array(waveform), ch, peak_index + self._sample)
                all_waveforms = self.extract_all(self._processor.circular_buffer[:8, :], peak_index)
                pk_id = self.append(ch, all_waveforms)

                # Add new spike ID to metadata
                self.__spike_metadata.append((peak_index, pk_id))

    def add_muap(self, new_muap_filter: "np.ndarray[np.float64]"):
        """
        Append a new MUAP filter to _muap_filters.
        """
        if self._muap_filters[self._state] is None:
            self._muap_filters[self._state] = new_muap_filter[np.newaxis, :]  # Initialize with first filter
        else:
            self._muap_filters[self._state] = np.vstack((self._muap_filters[self._state], new_muap_filter))  # Append new filter

        self._has_muaps = True  # Set the flag to indicate we have MUAP filters

    def handle_manual_covariance_setter(self):
        """
        Manually set the covariance matrix.
        """
        self.set_covariance()

    def handle_manual_clustering(self):
        self._has_muaps = False
        self._muap_filters[self._state] = None
        self.cluster()

    def _detect_threshold_crossings(self, data, threshold):
        """
        Detect threshold crossings in specified channel data and return peak indices.
        """
        threshold_crossings = []
        above_threshold = False

        for i in range(self._pre_peak_samples, len(data) - self._post_peak_samples):
            if (data[i] > threshold if threshold > 0 else data[i] < threshold):
                if not above_threshold:
                    above_threshold = True
                    peak_index = i
            elif above_threshold:
                # End of threshold crossing, identify peak in this segment
                above_threshold = False
                peak_index = np.argmax(data[peak_index:i]) + peak_index
                threshold_crossings.append(peak_index)
        return threshold_crossings

    def append(self, ch: int, waveform):
        """Buffers the waveform and metadata, triggering clustering if buffer is full."""
        # max_abs_value = np.max(np.abs(waveform))
        # if max_abs_value > 0:  # Avoid division by zero
        #     waveform = waveform / max_abs_value
        
        self._spike_buffer[self._spike_buffer_index] = waveform 
        self._spike_buffer_index = (self._spike_buffer_index + 1) % self._spike_buffer_size
        self._spike_buffer_full = self._spike_buffer_full or (self._spike_buffer_index == 0)

        return ch
    
    def extract(self, data: np.ndarray["float"], peak_index: int) -> "np.ndarray[np.float64]":
        """
        Extracts a waveform centered around the peak index.
        """
        return data[peak_index - self._pre_peak_samples : peak_index + self._post_peak_samples + 1]

    def extract_all(self, data, peak_index) -> "np.ndarray[np.float64]":
        """
        Extracts a 16-sample waveform centered around the peak index.
        
        Parameters:
        - data: 1D array of channel data from circular_buffer
        - peak_index: Index of the peak sample
        
        Returns:
        - 1D numpy array of the waveform (16 samples)
        """
        # Extract waveform as 10 samples before, peak, and 5 samples after
        return data[:,peak_index - self._pre_peak_samples: peak_index + self._post_peak_samples+1]

    def start_recording(self, fname):
        self._recording = True
        self._logger = SpikeLogger(fname)
        self._logger_file = fname

    def stop_recording(self):
        if not self._recording:
            return
        self._recording = False
        s = SpikeReader(self._num_ipts, self._logger_file)
        s.convert()
        s.close()
        del s
        self._logger_file = None

    def set_spike_buffering_state(self, state: bool) -> None:
        self._collecting_spike_buffer = state

    def set_covariance_buffering_state(self, state: bool) -> None:
        self._collecting_covariance_buffer = state

    @abstractmethod
    def _muaps(self) -> "np.ndarray[np.float64]":
        """
        Return the motor unit trains based on handling.
        """
        raise(NotImplementedError("_muaps is not implemented in BaseSpikeHandler. It should be overloaded in a subclass instead."))

    @abstractmethod
    def set_covariance(self):
        """
        Compute 'extended' sample covariance.
        (e.g. extension_factor of 16 --> covariance: 128 x 128 matrix from _emg_history buffer).
        Recovers covariance R of extended samples array, then sets
        self._P, the pseudo-inverse of this covariance matrix.
        """
        raise(NotImplementedError("set_covariance method is not implemented in BaseSpikeHandler. It should be overloaded in a subclass instead."))

    def set_verbose(self, verbose: bool):
        self._verbose = verbose

    def set_state(self, state: int):
        self._state = state
        self._has_muaps = (self._P[state] is not None) and (self._muap_filters[state] is not None)

    def set_adaptive(self, state_adaptation_enabled: bool):
        self._enable_state_adaptation = state_adaptation_enabled

    def count(self, min_spike_id=0):
        # Filter spikes that occurred within the last 25 samples (spike_id > 230)
        recent_spikes = [(spike_id, ch) for (spike_id, ch) in self.__spike_metadata if spike_id > min_spike_id]
        # Count occurrences per cluster index
        cluster_counts = Counter(ch for (_, ch) in recent_spikes)
        # Ensure a count entry for each cluster, setting default count to 0
        return [float(float(cluster_counts.get(cluster, 0)) / 0.50) for cluster in range(self._num_ipts)]

    def set_ipt_threshold_source(self, source: int): 
        self._source = source
        self._ipt_threshold = self._threshold[self._state][source].flatten() 
        # print(f"IPT Source set to index-{self._source}")

    def set_ipt_threshold_scalar(self, scalar: float):
        self._scalar = scalar
        self._threshold[self._state] = np.full_like(self._ipt_max_values[self._state],self._scalar)
        self._limit[self._state] = np.full_like(self._ipt_max_values[self._state], self._scalar+2*self._precision)
        self._ipt_threshold = self._threshold[self._state][self._source].flatten()

    @pyqtSlot()
    def set_max_values(self):
        ipt_mu = np.mean(np.abs(self._ipt_buffer), axis=1)
        self._ipt_max_values[self._state][:,-1] = np.where(ipt_mu > 1e-3, ipt_mu, 1e-3)
        self._threshold[self._state] = np.full_like(self._ipt_max_values[self._state],self._scalar)
        self._ipt_threshold = self._threshold[self._state][self._source].flatten() 

    def set_num_ipts(self, n: int):
        """Sets number of IPTs, resetting current rate estimates."""
        self._num_ipts = n
        self._ipt_max_values = np.ones((
            self._num_ipts, 1
        )).astype(np.float64)
        self._ipt_buffer = np.zeros((
            self._num_ipts, self._ipt_buffer_size
        )).astype(np.float64)
        self._ipt_normalized_buffer = np.zeros((
            self._num_ipts, self._ipt_buffer_size
        )).astype(np.float64)
        self._ipt_max_values = [np.ones((self._num_ipts,1)).astype(np.float64) ] * self._max_states
        self._threshold = [np.full((self._num_ipts, 1), self._scalar)] * self._max_states
        self._limit = [np.full((self._num_ipts, 1), 3*self._scalar)] * self._max_states

        self._rates = np.zeros(self._num_ipts, dtype=np.float32)
        # Also make sure to update the size of the processor rates buffer:
        self._processor._rates_x = np.ones(self._num_ipts+1, dtype=np.float32)
        self._initialize_filter() # Need to reset the filter states for spikes as well.
        if self._has_muaps:
            self.handle_manual_clustering()
        self.num_ipts_changed.emit(self._num_ipts)

    def rates(self):
        """Returns current rate estimates for each IPT."""
        return self._rates
    
    @staticmethod
    def _extend(data, extension_factor, buffer_size, num_channels: int = 8) -> "np.ndarray":
        extended_data = np.zeros((num_channels * extension_factor, buffer_size - extension_factor + 1))
        for i in range(extension_factor):
            # Shift data by `i` samples and assign to the corresponding section in extended_data
            extended_data[i * num_channels:(i + 1) * num_channels, :] = data[:, i:buffer_size - extension_factor + i + 1]
        return extended_data
    
    @pyqtSlot()
    def save(self):
        """Save the relevant attributes to a file."""
        data = {
            "_max_states": self._max_states,
            "max_amp": self.max_amp, 
            "_P": self._P,
            "_R": self._R,
            "_spike_threshold": self._spike_threshold,
            "_muap_filters": self._muap_filters,
            "_ipt_max_values": self._ipt_max_values,
            "_threshold": self._threshold,
            "_limit": self._limit,
            "_state_orientation": self._state_orientation,
            "_state_activity": self._state_activity, 
            "_scalar": self._scalar, 
            "_source": self._source, 
            "_state": self._state,
            "_precision": self._precision, 
            "_ipt_threshold": self._ipt_threshold, 
            "_num_ipts": self._num_ipts, 
            "_pre_peak_samples": self._pre_peak_samples, 
            "_post_peak_samples": self._post_peak_samples, 
            "_state_likelihood_alpha": self._state_likelihood_alpha
        }
        with open(self._model_file, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Saved model state to {self._model_file}")

    @pyqtSlot()
    def load(self):
        """Load the relevant attributes from a file."""
        if not os.path.exists(self._model_file):
            print(f"⚠️ Load failed: {self._model_file} not found.")
            return
        
        with open(self._model_file, "rb") as f:
            data = pickle.load(f)

        if "_pre_peak_samples" in data:
            self._pre_peak_samples = data["_pre_peak_samples"]
            self._post_peak_samples = data["_post_peak_samples"]
            self._extension_factor = self._pre_peak_samples + self._post_peak_samples + 1
            self.set_num_ipts(data["_num_ipts"])
        
        # Restore attributes
        self.set_max_states(int(data["_max_states"]))
        
        if len(data["max_amp"]) < self._max_states:
            print(f"Less than {self._max_states} states found in model. Loading only available states.")
            for (i, P) in enumerate(data["max_amp"]):
                print(f" -> Loaded state-{i}")
                self._P[i] = P
                self._R[i] = data["_R"][i]
                self.max_amp[i] = data["max_amp"][i]
                self._muap_filters[i] = data["_muap_filters"][i]    
                self._ipt_max_values[i] = data["_ipt_max_values"][i]
                self._threshold[i] = data["_threshold"][i]
                self._limit[i] = data["_limit"][i]
                self._spike_threshold[i] = data["_spike_threshold"][i]
                self._state_orientation[i] = data["_state_orientation"][i]
                self._state_activity[i] = data["_state_activity"][i]
        else:
            self._P = data["_P"]
            self._R = data["_R"]
            self.max_amp = data["max_amp"]
            self._muap_filters = data["_muap_filters"]
            self._ipt_max_values = data["_ipt_max_values"]
            self._threshold = data["_threshold"]
            self._limit = data["_limit"]
            self._spike_threshold = data["_spike_threshold"]
            self._state_orientation = data["_state_orientation"]
            self._state_activity = data["_state_activity"]
        self._precision = np.float64(data["_precision"])
        self._ipt_threshold = np.float64(data["_ipt_threshold"])
        self._scalar = np.float64(data["_scalar"])
        self._source = int(data["_source"])
        self._state = int(data["_state"])
        if "_state_likelihood_alpha" in data:
            self._state_likelihood_alpha = np.float64(data["_state_likelihood_alpha"])
        
        for index, R in enumerate(self._R):
            if R is None:
                self._spread[index] = 0
            else:
                self._spread[index] = 0.1 * np.log(np.trace(self._R[index]))
                print(f"Spread for state-{index}: {self._spread[index]:.2f}")

        self._has_muaps = (self._P[self._state] is not None) and (self._muap_filters[self._state] is not None)
        print(f"✅ Loaded model state from {self._model_file}")