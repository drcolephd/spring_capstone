import numpy as np
from numpy.typing import NDArray


class MlpWindowExtractor:
    """
    Sliding-window EMG feature extractor for MLP gesture recognition.

    Maintains a ring buffer and emits a feature vector every STRIDE_SAMPLES
    new samples once the window is full.

    4 features per channel (RMS, MAV, waveform length, variance).
    Defaults: 50-sample window (100 ms @ 500 Hz), 5-sample stride (10 ms).
    """

    WINDOW_SAMPLES: int = 50
    STRIDE_SAMPLES: int = 5
    FEATURES_PER_CH: int = 4

    def __init__(self, num_channels: int = 8):
        self.num_channels = num_channels
        self.feature_dim = num_channels * self.FEATURES_PER_CH
        self._buf = np.zeros((num_channels, self.WINDOW_SAMPLES), dtype=np.float32)
        self._write_pos: int = 0
        self._fill_count: int = 0
        self._since_last: int = 0

    def push_frame(self, frame: NDArray) -> "NDArray[np.float32] | None":
        """Push one sample frame [num_channels]. Returns feature vector or None."""
        slot = self._write_pos % self.WINDOW_SAMPLES
        self._buf[:, slot] = frame[:self.num_channels]
        self._write_pos += 1
        if self._fill_count < self.WINDOW_SAMPLES:
            self._fill_count += 1
        self._since_last += 1

        if self._fill_count < self.WINDOW_SAMPLES or self._since_last < self.STRIDE_SAMPLES:
            return None
        self._since_last = 0
        return self._compute_features()

    def push_batch(self, data: NDArray) -> "list[NDArray[np.float32]]":
        """Push batch [num_channels × N]. Returns all feature vectors emitted."""
        results = []
        for i in range(data.shape[1]):
            feat = self.push_frame(data[:, i])
            if feat is not None:
                results.append(feat)
        return results

    def _compute_features(self) -> "NDArray[np.float32]":
        start = self._write_pos % self.WINDOW_SAMPLES
        idx = (start + np.arange(self.WINDOW_SAMPLES)) % self.WINDOW_SAMPLES
        ordered = self._buf[:, idx].astype(np.float64)

        result = np.empty(self.feature_dim, dtype=np.float32)
        for ch in range(self.num_channels):
            x = ordered[ch]
            mean = x.mean()
            rms = float(np.sqrt(np.mean(x * x)))
            mav = float(np.mean(np.abs(x)))
            wl = float(np.sum(np.abs(np.diff(x))))
            var = float(np.mean((x - mean) ** 2))
            base = ch * 4
            result[base]     = rms
            result[base + 1] = mav
            result[base + 2] = wl
            result[base + 3] = var
        return result

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._write_pos = 0
        self._fill_count = 0
        self._since_last = 0
