import numpy as np
from nml.spikes.base import BaseSpikeHandler
from PyQt5.QtCore import pyqtSignal


class NEOSpikeHandler(BaseSpikeHandler):
    """
    Spike handler that computes the Nonlinear Energy Operator (NEO)
    and performs template matching in NEO space.

    Unlike BeamformerSpikeHandler, this handler:
      - Computes Ψ(x[n]) = x[n]^2 - x[n-1]*x[n+1]
      - Clusters based on NEO waveforms
      - Matches templates using NEO-projected segments
      - Emits embedding coordinates based on NEO-derived features
    """
    source_spike = pyqtSignal(object, int, float)
    neo_detected = pyqtSignal(object, int, int)  # (neo_waveform, channel, sample)
    
    def __init__(self,
                 processor,
                 buffer_size: int = 512,
                 pre_peak_samples: int = 8,
                 post_peak_samples: int = 8,
                 enable: bool = True,
                 compute_thresholds: bool = True,
                 num_ipts: int = 64,
                 channel: int = None,
                 threshold: float = None,
                 max_states: int = 10,
                 model_file: str = "neo_muaps_model.pkl"):

        super().__init__(processor,
                 buffer_size=buffer_size,
                 pre_peak_samples=pre_peak_samples,
                 post_peak_samples=post_peak_samples,
                 enable=enable,
                 compute_thresholds=compute_thresholds,
                 num_ipts=num_ipts,
                 channel=channel,
                 threshold=threshold,
                 max_states=max_states,
                 model_file=model_file)
        self._neo_spike_templates = None
        # Length of NEO streaming history buffer
        self._neo_history_size = 16384        
        # 8 EMG channels × neo-history buffer
        self._neo_buffer = np.zeros((8, self._neo_history_size), dtype=np.float64)

    # ---------------------------------------------------------
    # NEO computation
    # ---------------------------------------------------------
    @staticmethod
    def compute_neo(x: np.ndarray) -> np.ndarray:
        neo = x[1:-1]**2 - x[:-2] * x[2:]
        return np.pad(neo, (1,1), mode='edge')

    # ---------------------------------------------------------
    # NEO spike detection
    # ---------------------------------------------------------
    def _detect_neo_spikes(self, neo_data, num_samples):

        # Update mangled metadata buffer
        self._BaseSpikeHandler__spike_metadata = [
            (sp - num_samples, ch)
            for (sp, ch) in self._BaseSpikeHandler__spike_metadata
        ]
        self._BaseSpikeHandler__spike_metadata = [
            (sp, ch)
            for (sp, ch) in self._BaseSpikeHandler__spike_metadata
            if sp >= 0
        ]

        # Threshold detection per channel
        for ch in range(8):
            peak_indices = self._detect_threshold_crossings(
                neo_data[ch], self._spike_threshold[ch]
            )
            for peak_index in peak_indices:

                # Avoid duplicate spike IDs
                if not any(sp == peak_index for (sp, _) in self._BaseSpikeHandler__spike_metadata):

                    neo_waveform = self.extract(neo_data[ch], peak_index)
                    self.neo_detected.emit(neo_waveform, ch, peak_index + self._sample)

                    # Multi-channel NEO snippet
                    neo_all = self.extract_all(neo_data, peak_index)
                    muap_id = self.append(ch, neo_all)

                    self._BaseSpikeHandler__spike_metadata.append((peak_index, muap_id))

    def handle_spike_detection(self, num_samples: int):
        emg = self._processor.circular_buffer[:8, :]
        C, N = emg.shape

        neo = np.zeros((C, N))
        for ch in range(C):
            neo[ch] = NEOSpikeHandler.compute_neo(emg[ch])

        # Roll NEO buffer
        self._neo_buffer[:, :-num_samples] = self._neo_buffer[:, num_samples:]
        self._neo_buffer[:, -num_samples:] = neo[:, -num_samples:]

        # Run detection
        self._detect_neo_spikes(neo, num_samples)

    def handle_clustering(self):
        """
        NEO-based clustering:
        - Uses self._neo_buffer to detect suprathreshold activity.
        - Does not perform PCA or KMeans; clusters = channels.
        """
        if self._has_muaps:
            # Shift buffers
            self._ipt_buffer[:, :-1] = self._ipt_buffer[:, 1:]
            self._ipt_normalized_buffer[:, :-1] = self._ipt_normalized_buffer[:, 1:]

            # Latest NEO sample (8 channels)
            sample = self._neo_buffer[:, -1].reshape(-1, 1)

            # Store raw
            self._ipt_buffer[:, -1:] = sample

            # Normalize by per-IPT max value
            norm_sample = sample / (self._ipt_max_values[self._state] + 1e-6)
            self._ipt_normalized_buffer[:, -1:] = norm_sample

            # Boolean which IPTs fired based on scalar threshold
            threshold_crossed = (
                (norm_sample[:, 0] > self._threshold[self._state][:, 0]) &
                (norm_sample[:, 0] < self._limit[self._state][:, 0])
            )

            # Emit source spike only from selected IPT
            if self._source_mode and threshold_crossed[self._source]:
                self.source_spike.emit(
                    sample,
                    self._sample,
                    self._processor.wrist_orientation[0]
                )

    # ---------------------------------------------------------
    # NEO clustering
    # ---------------------------------------------------------
    def cluster(self):
        if not self._spike_buffer_full:
            print("Buffer not full; cannot perform NEO clustering.")
            return

        reshaped = self._spike_buffer.reshape(self._spike_buffer_size, 8, -1)
        neo_p2p = np.ptp(reshaped, axis=2)

        norm = neo_p2p / np.max(neo_p2p, axis=0, keepdims=True)

        embedding = np.column_stack([
            np.mean(norm[:, :4], axis=1),
            np.mean(norm[:, 4:], axis=1)
        ])

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self._num_ipts)
        labels = km.fit_predict(embedding)

        templates = []
        for k in range(self._num_ipts):
            mean_wf = reshaped[labels == k].mean(axis=0)
            templates.append(mean_wf.flatten())

        self._neo_spike_templates = np.array(templates)
        self._muap_filters[self._state] = self._neo_spike_templates
        self._has_muaps = True

        print("NEO MUAP templates acquired.")

        self.set_ipt_threshold_scalar(self._scalar)

    # ---------------------------------------------------------
    # Project single-sample NEO patch onto templates
    # ---------------------------------------------------------
    def _muaps(self):

        if self._neo_spike_templates is None:
            return np.zeros((self._num_ipts, 1))

        # Use only the last extension_factor samples
        latest = self._neo_buffer[:, -self._extension_factor:]   # (8, 16)
        extended = latest.flatten().reshape(-1, 1)                # (128, 1)

        return np.abs(self._neo_spike_templates @ extended)
