import numpy as np
import csv
from scipy.ndimage import convolve
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FrequencyBuffer:
    _threshold_fraction = 0.5
    _angle = [np.pi/2, 0, 3*np.pi/2, np.pi]
    classifier = None

    def __init__(self, num_channels, num_freq_bands, capacity, threshold_fraction = None):
        """
        Initialize the FrequencyBuffer to store frequency magnitude data.
        
        Parameters:
        - num_channels: int, number of EMG channels
        - num_freq_bands: int, number of frequency bands per channel
        - capacity: int, the maximum number of frequency windows to store
        - threshold_fraction: float, scalar used in calculation in `get_threshold()` method (default: 0.5)
        """
        self.num_channels = num_channels
        self.num_freq_bands = num_freq_bands
        self._capacity = capacity
        self._active = np.zeros((self.num_channels, self.num_freq_bands))
        self.baseline = np.zeros((self.num_channels, self.num_freq_bands))
        self.threshold = np.full(self.num_channels, self._threshold_fraction)
        self._label = np.full(capacity, -1)
        self._angle = np.zeros((self.num_channels, capacity))
        if threshold_fraction is not None:
            self._threshold_fraction = threshold_fraction
        
        # Buffer for each channel, shape: (num_channels, capacity, num_freq_bands)
        self.buffer = np.zeros((num_channels, capacity, num_freq_bands))
        
        # Track the current index for each channel
        self.index = np.zeros(num_channels, dtype=int)
        self._label_index = 0
        self._angle_index = 0
        self.full = np.zeros(num_channels, dtype=bool)  # Track if each channel's buffer is full

    def append(self, channel, freq_magnitude):
        """
        Append frequency magnitude data to the buffer for a specific channel.
        
        Parameters:
        - channel: int, the channel index
        - freq_magnitude: array-like, frequency magnitude data for the channel
        """
        # Ensure freq_magnitude has the correct size
        if len(freq_magnitude) != self.num_freq_bands:
            raise ValueError(f"Expected {self.num_freq_bands} frequency bands, got {len(freq_magnitude)}")
        
        # Insert the frequency magnitudes at the current index for the specified channel
        self.buffer[channel, self.index[channel]] = freq_magnitude
        
        # Update index and check if buffer is full
        self.index[channel] = (self.index[channel] + 1) % self._capacity
        if self.index[channel] == 0:
            self.full[channel] = True

    def add_label(self, val):
        """
        Append label to the buffer for the current sample. 
        """
        self._label[self._label_index] = val
        self._label_index = (self._label_index + 1) % self._capacity

    def add_angle(self, theta):
        """"""
        self._angle[:,self._angle_index] = theta
        self._angle_index = (self._angle_index + 1) % self._capacity

    def _flush(self):
        """
        Reset the buffers and remove flag indicating they are full.
        """
        self.buffer = np.zeros((self.num_channels, self._capacity, self.num_freq_bands))
        self._label = np.full(self._capacity, -1)
        self._label_index = 0
        self._angle = np.zeros((self.num_channels, self._capacity))
        self._angle_index = 0
        for channel in range(self.num_channels):
            self.full[channel] = False
            self.index[channel] = 0

    def estimate_baseline(self):
        """
        Calculate the mean baseline frequency response for each channel.
        """
        for channel in range(self.num_channels):
            if self.full[channel]:
                # Use full buffer
                self.baseline[channel] = np.mean(self.buffer[channel], axis=0)
            else:
                # Use only filled portion of the buffer
                self.baseline[channel] = np.mean(self.buffer[channel, :self.index[channel]], axis=0)

    def estimate_threshold(self):
        """
        Calculate the threshold as a fraction of the mean intensity for each channel.
        """
        for channel in range(self.num_channels):
            if self.full[channel]:
                data = self.buffer[channel]
            else:
                data = self.buffer[channel, :self.index[channel]]
            self._active[channel] = np.mean(data, axis=0) - self.baseline[channel]
            self.threshold[channel] = self._threshold_fraction * np.trapz(np.where(self._active[channel] > 0, self._active[channel], 0))
        
    def export_features(self, filename):
        """
        Export the labeled dataset.

        Parameters:
        - filename: str, the name of the CSV file to save the features and labels.
        """
        # Open the file for writing
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header row
            header = ['label', 'ch'] + [f'feature_{i}' for i in range(self.num_freq_bands)]
            writer.writerow(header)
            
            # Iterate over each labeled sample in the buffer
            for i in range(self._capacity):
                label = self._label[i]
                adjusted_magnitudes = self.buffer[:, i, :]
                for ch in range(8):
                    sample_features = adjusted_magnitudes[ch]
                    # Write label and features to the CSV
                    row = [label, ch] + sample_features.tolist()
                    writer.writerow(row)

        print(f"Features and labels exported to {filename}")

    def is_full(self) -> bool:
        """
        Check if all channels are full and return True if yes, False if no.
        """
        all_full = True
        for channel in range(self.num_channels):
            all_full = all_full and self.full[channel]
        return all_full
    
    def predict_direction(self, sample):
        """
        Predict the direction for a new sample using the trained classifier.
        
        Parameters:
        - sample: 2D numpy array (num_channels, num_freq_bands), adjusted frequency magnitudes.
        
        Returns:
        - Predicted direction label.
        """
        if self.classifier is None:
            return -1

        # Apply convolutions to each channel's adjusted magnitudes and concatenate results
        sample_features = []
        for channel_data in sample:
            channel_features = self.apply_convolutions(channel_data)
            sample_features.extend(channel_features)

        # Convert to numpy array and reshape for classifier input
        sample_features = np.array(sample_features).reshape(1, -1)

        # Predict the direction
        predicted_label = self.classifier.predict(sample_features)
        return predicted_label[0]
    
    def set_capacity(self, value):
        """
        Set a new value for the capacity (number of windows that the buffer holds).
        """
        self._capacity = value
        self._flush()

    def set_threshold_fraction(self, value):
        """
        Set a new value for the gain on the thresholds computed in the get_threshold method.
        """
        self._threshold_fraction = value
        print(f"Updated threshold scalar to {value} (FrequencyBuffer: RealTimePlotter.calibration_buffer)")

    def clear(self):
        """
        Clear the buffer data, resetting all indices and 'full' flags.
        """
        self.buffer.fill(0)
        self.index.fill(0)
        self.full.fill(False)

    @staticmethod
    def apply_convolutions(sample):
        """
        Apply a set of convolutional kernels to a sample and return flattened feature vector.
        """
        kernels = [
            np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),  # Vertical edge
            np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),  # Horizontal edge
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  # Center-surround
        ]
        features = []
        for kernel in kernels:
            convolved = convolve(sample, kernel, mode='reflect')
            # Pooling (e.g., max pooling over the entire convolved "image")
            pooled_feature = np.max(convolved)
            features.append(pooled_feature)
        return np.array(features)
