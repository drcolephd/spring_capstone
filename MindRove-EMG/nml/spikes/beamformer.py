from nml.spikes.base import BaseSpikeHandler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from typing import Optional, Tuple
from nml.priors import electrode_angles

class BeamformerSpikeHandler(BaseSpikeHandler):
    source_spike = pyqtSignal(object, int, float)
    _electrode_angles = electrode_angles
    _source_mode: bool = False

    _in_ipt_debounce = None
    _ipt_debounce_counter = None
    _ipt_debounce_threshold: int = None

    def __init__(self, processor, buffer_size: int = 512, 
                 pre_peak_samples: int = 6,
                 post_peak_samples: int = 9, 
                 enable: bool = True, 
                 compute_thresholds: bool = True, 
                 num_ipts: int = 64, 
                 threshold: float = None, 
                 channel: int = None, 
                 max_states: int = 10, 
                 model_file: str = "max_muaps_model.pkl" ):
        super().__init__(processor, 
                 buffer_size = buffer_size, 
                 pre_peak_samples = pre_peak_samples,
                 post_peak_samples = post_peak_samples, 
                 enable = enable, 
                 compute_thresholds = compute_thresholds, 
                 num_ipts = num_ipts, 
                 threshold=threshold, 
                 channel=channel, 
                 max_states=max_states, 
                 model_file=model_file)
        self._in_ipt_debounce = np.zeros(self._num_ipts, dtype=bool)  # One debounce flag per channel
        self._ipt_debounce_counter = np.zeros(self._num_ipts, dtype=int)  # Debounce counters
        self._ipt_debounce_threshold = self._post_peak_samples + 1

    def append(self, ch, waveform: "np.ndarray[np.float64]") -> int:
        """Buffers the waveform and metadata, triggering clustering if buffer is full."""

        self._spike_buffer[self._spike_buffer_index] = waveform 
        self._spike_buffer_index = (self._spike_buffer_index + 1) % self._spike_buffer_size
        self._spike_buffer_full = self._spike_buffer_full or (self._spike_buffer_index == 0)

        # Ensure MUAP filters are properly reshaped for projection
        if self._muap_filters[self._state] is not None:
            if self._muap_filters[self._state].shape[-1] != waveform.size:
                self._muap_filters[self._state] = self._muap_filters[self._state].reshape(self._muap_filters[self._state].shape[0], -1)

            # Project waveform onto each MUAP template and find best match
            projections = np.dot(self._muap_filters[self._state], waveform.flatten())
            id = np.argmax(projections)  # Best matching cluster
        else:
            # If no MUAP templates are available, use channel for cluster assignment
            id = ch
        p2p_amplitudes = 0.35 * np.ptp(waveform/np.max(waveform), axis=1)
        # print(p2p_amplitudes)
        embedding_x = np.dot(p2p_amplitudes,np.cos(self._electrode_angles))
        embedding_y = np.dot(p2p_amplitudes,np.sin(self._electrode_angles))

        # Emit suprathreshold event with cluster assignment and embedding coordinates
        self.embedding.emit(
            embedding_x, embedding_y, self._sample, ch, self._processor.wrist_orientation[0], id
        )

        if self._spike_buffer_full and self._needs_initial_clustering:
            self._needs_initial_clustering = False
            self.cluster()
            self.set_covariance()
            self.set_ipt_threshold_scalar(self._scalar)
        return id
    
    def enable_source_mode(self, enable: bool = True):
        """Enables or disables source mode for the spike handler."""
        self._source_mode = enable
    
    def _compute_state_likelihoods(self, x, x_env, activity, orientation):
        """Computes log-likelihood of a sample given covariance, IPT out-of-bounds, orientation, and activity."""
        likelihoods = np.zeros(self._max_states)

        for index, P in enumerate(self._P):
            if P is None:
                likelihoods[index] = self._lb_log_likelihood
            else:
                ### 1️⃣ Compute Mahalanobis Distance Term (Whitened EMG Distance)
                mahalanobis = 0.05 * np.log(np.dot(x_env.T, np.dot(P, x_env))).flatten()[0]

                ### 2️⃣ Compute Out-of-Bounds IPT Power Term
                ipt_values = self._muap_filters[index] @ P @ x  # Projected IPT power
                lower_bound_violations = np.maximum(0, self._threshold[index] - ipt_values)
                upper_bound_violations = np.maximum(0, ipt_values - self._limit[index])
                ipt_penalty = 0.1 * np.log10(0.1 * np.sum(lower_bound_violations + upper_bound_violations) + 1)  

                ### 3️⃣ Compute Gyro Orientation Similarity Term
                current_orientation = orientation / (np.linalg.norm(orientation) + 1e-12)  # Normalize input
                stored_orientation = self._state_orientation[index].copy() / (np.linalg.norm(self._state_orientation[index]) + 1e-12)  # Normalize stored
                cos_sim = np.dot(stored_orientation, current_orientation)
                gyro_likelihood = np.arctan(cos_sim - 0.5)  # Penalizes large orientation mismatch

                ### 4️⃣ Compute Accelerometer Power Similarity Term
                activity_distance = 0.05 * np.sqrt(np.abs(activity - self._state_activity[index]))  # Penalize activity deviation

                ### 🔥 Combine the Likelihood Terms
                likelihoods[index] = max(mahalanobis + gyro_likelihood - activity_distance - ipt_penalty, self._lb_log_likelihood)

                if self._verbose:
                    print(f"likelihood[{index}] = {likelihoods[index]:.2f}: {mahalanobis:.2f} + {gyro_likelihood:.2f} - {activity_distance:.2f} - {ipt_penalty:.2f}")
        return likelihoods

    def _update_state(self, x, x_env):
        """Updates state estimation based on streaming data."""
        
        # Compute log-likelihoods for all states
        log_likelihoods = self._compute_state_likelihoods(x, x_env, self._history_acc, self._history_orientation)

        # Apply exponential moving average smoothing to likelihoods
        self._state_likelihoods = (
            self._state_likelihood_alpha * self._state_likelihoods 
            + (1 - self._state_likelihood_alpha) * log_likelihoods
        )

        # Select the state with the maximum smoothed likelihood
        self._state = np.argmax(self._state_likelihoods)

    def set_ipt_threshold_scalar(self, scalar: float):
        self._scalar = scalar

        # 1. Extend _emg_history by creating delayed versions
        num_channels, buffer_size = self._emg_history.shape
        extended_history = self._extend(self._emg_history, self._extension_factor, buffer_size, num_channels)

        # 2. Project the extended history using whitening and muap filters
        ipt_history = self._muap_filters[self._state] @ (self._P[self._state] @ extended_history)

        # 3. Normalize rows of `ipt_history` by the max value in each row (which is saved in `self._ipt_max_values`)
        ipt_mu = np.mean(np.abs(self._ipt_buffer), axis=1)  # (self._num_ipts,)
        self._ipt_max_values[self._state][:, -1] = np.where(ipt_mu > 1e-3, ipt_mu, 1e-3)  # Ensure proper shape
        ipt_history /= self._ipt_max_values[self._state][:, -1][:, np.newaxis]  # Normalize column-wise

        # 4. Compute the ideal threshold and upper-bound for each row of `ipt_history` 
        # Use masked selection to avoid flattening
        # mask = ipt_history > self._scalar
        # masked_ipt = np.where(mask, ipt_history, np.nan)  # Replace non-valid values with NaN
        # ipt_mu = np.nanmean(masked_ipt, axis=1)  # Compute mean while ignoring NaNs
        # ipt_sigma = 0.5 * np.nanstd(masked_ipt, axis=1)  # Standard deviation for range
        ipt_mu = np.mean(ipt_history, axis=1)

        # Assign threshold and limit with correct shape
        self._threshold[self._state][:, -1] = np.where(ipt_mu - self._precision > self._scalar, ipt_mu - self._precision, self._scalar)
        self._limit[self._state][:, -1] = self._threshold[self._state][:,-1] +  2*self._precision
        self._ipt_threshold = self._threshold[self._state][self._source].flatten()

    def set_precision(self, precision: float):
        """
        Sets self._precision to a value between self._precision_best and self._precision_worst based on the input precision.
        """
        self._precision = max(self._precision_worst, min(self._precision_best, precision))

    def cluster(self):
        """Perform PCA and K-means clustering to identify spike clusters."""
        if not self._spike_buffer_full:
            print("Buffer not full, cannot perform clustering.")
            return

        reshaped_spikes = self._spike_buffer.reshape(self._spike_buffer_size, 8, -1)  # (buffer_size, 8, waveform_length)
    
        # Compute peak-to-peak amplitudes (p2p)
        p2p_amplitudes = np.ptp(reshaped_spikes, axis=2)  # Shape: (buffer_size, 8)

        # Compute 2D embeddings
        norm_p2p = p2p_amplitudes / self.max_amp[self._state].reshape(1,8)  # Normalize by max amplitude
        embedding_x = np.dot(norm_p2p, np.cos(self._electrode_angles)) * 0.5  # Shape: (buffer_size,)
        embedding_y = np.dot(norm_p2p, np.sin(self._electrode_angles)) * 0.5  # Shape: (buffer_size,)
        # Compute principal eigenvalue of waveform covariance for each spike
        principal_eigenvalues = np.zeros(self._spike_buffer_size)
        for i in range(self._spike_buffer_size):
            cov_matrix = np.cov(reshaped_spikes[i])  # Compute covariance matrix (8x8)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)  # Get sorted eigenvalues
            principal_eigenvalues[i] = eigenvalues[-1]  # Take the largest eigenvalue

        # Normalize principal eigenvalues
        principal_eigenvalues /= np.max(principal_eigenvalues)

        # Construct feature matrix: (embedding_x, embedding_y, principal_eigenvalue)
        features = np.column_stack((embedding_x, embedding_y, principal_eigenvalues))  # Shape: (buffer_size, 3)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self._num_ipts)
        cluster_labels = kmeans.fit_predict(features)

        # Calculate cluster templates
        cluster_templates = []
        for cluster_id in range(self._num_ipts):
            cluster_waveforms = self._spike_buffer[cluster_labels == cluster_id]
            mean_waveform = np.mean(cluster_waveforms, axis=0) / self.max_amp[self._state]
            cluster_templates.append(mean_waveform.flatten())  # Ensure correct shape

        self._muap_filters[self._state] = np.array(cluster_templates)
        self._has_muaps = self._P[self._state] is not None
        self._threshold[self._state] = np.ones((self._num_ipts,1), dtype=np.float32) * self._scalar
        self._ipt_threshold = self._threshold[self._state][self._source].flatten()
        print("New MUAP filters acquired.")
        if self._has_muaps:
            self.set_ipt_threshold_scalar(self._scalar)

    @staticmethod
    def rotate_embedding(embedding_x: float, embedding_y: float, orientation: float) -> Tuple[float, float]:
        # Calculate the cosine and sine of the orientation angle
        cos_orientation = -np.cos(orientation)
        sin_orientation = np.sin(orientation)
        
        # Define the rotation matrix
        rotation_matrix = np.array([
            [cos_orientation, -sin_orientation],
            [sin_orientation, cos_orientation]
        ])
        
        # Apply the rotation to the embedding
        rotated_position = np.dot(rotation_matrix, np.array([embedding_x, embedding_y]))
        rotated_x, rotated_y = rotated_position[0], rotated_position[1]
        
        return rotated_x, rotated_y

    def set_covariance(self):
        """
        Compute 'extended' sample covariance and set the ZCA whitening matrix.
        """
        self._state_orientation[self._state] = self._history_orientation
        self._state_activity[self._state] = self._history_acc

        num_channels, buffer_size = self._emg_history.shape

        max_amps = np.max(np.abs(self._emg_history), axis=1)
        self.max_amp[self._state][:, :] = max_amps.reshape(-1, 1)

        extended_data = self._extend(self._emg_history, self._extension_factor, buffer_size, num_channels)

        # Covariance
        self._R[self._state] = np.cov(extended_data)

        # ZCA whitening: U @ D^{-1/2} @ U.T
        U, S, _ = np.linalg.svd(self._R[self._state])
        epsilon = 1e-5
        D_inv_sqrt = np.diag(1.0 / np.sqrt(S + epsilon))
        self._P[self._state] = U @ D_inv_sqrt @ U.T

        n, m = self._P[self._state].shape
        self._has_muaps = self._muap_filters[self._state] is not None

        print(f"Recovered {n} x {m} ZCA whitening projection matrix.")

        if self._has_muaps:
            self.set_ipt_threshold_scalar(self._scalar)


    def _muaps(self) -> "np.ndarray[np.float64]":
        """
        Compute motor unit action potentials (MUAPs) projection onto learned filters.

        Returns:
            np.ndarray[np.float64]: A (num_ipts x 1) vector representing the current sample IPT projection.
        """
        # Ensure _muap_filters is correctly reshaped to (num_ipts, nChannels * nExtendedSamples)
        if self._muap_filters[self._state].shape[1:] == (8, self._extension_factor):  
            self._muap_filters[self._state] = self._muap_filters[self._state].reshape(self._muap_filters[self._state].shape[0], -1)

        # Flatten _current_buffer to match (nChannels * nExtendedSamples, 1)
        extended_vector = self._current_buffer.flatten().reshape(-1, 1)  # Shape (128, 1)

        # Apply whitening transformation: (128, 128) @ (128, 1) → (128, 1)
        transformed_vector = self._P[self._state] @ extended_vector

        # Project onto MUAP filters: (num_ipts, 128) @ (128, 1) → (num_ipts, 1)
        return self._muap_filters[self._state] @ np.abs(transformed_vector)
    
    def muaps(self) -> "np.ndarray[np.float64]":
        return self._ipt_normalized_buffer[:, -1:]

    def handle_clustering(self):
        if self._has_muaps:
            self._ipt_buffer[:, :-1] = self._ipt_buffer[:, 1:]
            self._ipt_normalized_buffer[:, :-1] = self._ipt_normalized_buffer[:, 1:]
            sample = self._muaps()
            self._ipt_buffer[:, -1:] = sample
            self._ipt_normalized_buffer[:, -1:] = sample.copy() / self._ipt_max_values[self._state]

            # Emit only from channels that crossed the threshold and are NOT debounced
            self._ipt_normalized_buffer[self._in_ipt_debounce, -1] = 0
            threshold_crossed = (self._ipt_normalized_buffer[:, -1] > self._threshold[self._state]) & (self._ipt_normalized_buffer[:, -1] < self._limit[self._state])
            active_channels = np.where(threshold_crossed & ~self._in_ipt_debounce)[0]
            if active_channels.size > 0:
                self._in_ipt_debounce[active_channels] = True  # Mark channels as debounced
                if self._source_mode and self._ipt_normalized_buffer[self._source, -1] > self._threshold[self._state][self._source] and self._ipt_normalized_buffer[self._source, -1] < self._limit[self._state][self._source]:
                    # whitened_buffer = self._P[self._state] @ (self._current_buffer.copy().flatten().reshape(-1,1))
                    self.source_spike.emit(self._current_buffer.copy().flatten().reshape(-1,1), self._sample, self._processor.wrist_orientation[0])

            # Update debounce counters for all channels
            self._ipt_debounce_counter += 1

            # Reset debounce for channels that have completed debounce period
            reset_channels = np.where(self._ipt_debounce_counter >= self._ipt_debounce_threshold)[0]
            self._ipt_debounce_counter[reset_channels] = 0
            self._in_ipt_debounce[reset_channels] = False
            if self._enable_state_adaptation:
                x_env = self._current_envelope.flatten().reshape(-1,1)
                x = self._current_buffer.flatten().reshape(-1,1)
                self._update_state(x, x_env)

    def save_muap_filters_and_thresholds(self, file_path: Optional[str] = None) -> None:
        """
        Save _muap_filters and _spike_threshold to separate CSV files.
        The threshold file name is derived from the MUAP filter file name.

        Parameters:
        - file_path (Optional[str]): Base path to save the CSV files. If None, a file dialog is used.
        """
        if self._muap_filters[self._state] is None:
            print("No MUAP filters to save.")
            return

        # If no file path is specified, prompt the user to select one
        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                None, "Save MUAP Filters", "models/_muap_filters.csv", "CSV Files (*.csv);;All Files (*)"
            )

        # Proceed if a file path was selected or provided
        if file_path:
            # Save MUAP filters to the specified file path
            np.savetxt(file_path, self._muap_filters[self._state], delimiter=",", fmt="%.6e")
            print(f"MUAP filters saved to {file_path}")

            # Derive threshold file path by appending "_emg_thresholds.csv"
            base_path = file_path.removesuffix("_muap_filters.csv")
            threshold_file_path = f"{base_path}_emg_thresholds.csv"
            
            # Save spike thresholds
            np.savetxt(threshold_file_path, self._spike_threshold[np.newaxis, :], delimiter=",", fmt="%.6e")
            print(f"EMG thresholds saved to {threshold_file_path}")
        else:
            print("Save canceled.")

    def load_models(self, rates_model_path: str = None) -> None:
        """
        Load the _rates_model, _muap_filters, and _spike_thresholds attributes from
        CSV files with the same base name but different suffixes.

        Parameters:
        - rates_model_path (Optional[str]): Base filename (ends with _rates_model.csv) for loading. 
        If None, a file dialog is used to select the file.
        """
        # If no base filename is provided, prompt for a model file in the ~/models directory
        if rates_model_path is None:
            rates_model_path, _ = QFileDialog.getOpenFileName(
                None, "Select Model File", "models/_rates_model.csv", "Rate Models (*_rates_model.csv);;All Files (*);; CSV Files (*.csv)"
            )
            if rates_model_path is None:
                print("Canceled load.")
                return

        # Define paths for each required file based on the base filename
        muap_filters_path = rates_model_path.replace("_rates_model.csv", "_muap_filters.csv")
        emg_thresholds_path = rates_model_path.replace("_rates_model.csv", "_emg_thresholds.csv")
        if not (os.path.exists(rates_model_path) and os.path.exists(muap_filters_path) and os.path.exists(emg_thresholds_path)):
            print(f"Not all files (_rates_model.csv, _muap_filters.csv, and _emg_thresholds.csv) exist for {rates_model_path}")
            print("Model, filters, and thresholds not loaded.")
            return
        self._rates_model = np.loadtxt(rates_model_path, delimiter=",", skiprows=2)
        self._muap_filters[self._state] = np.loadtxt(muap_filters_path, delimiter=",")
        self._spike_threshold = np.loadtxt(emg_thresholds_path, delimiter=",").flatten()
        self._has_rates_model = True
        self._has_muaps = True
        self._needs_initial_clustering = False
