# nml/processing/hpf.py
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi


class HighPassFilter:
    """
    @brief Multi-channel high-pass filter (SOS) with persistent filter state.

    @param num_channels
        Number of channels to filter.
    @type num_channels: int

    @param fs
        Sampling frequency in Hz.
    @type fs: float

    @param cutoff_hz
        High-pass cutoff frequency in Hz.
    @type cutoff_hz: float

    @param order
        Filter order.
    @type order: int
    """

    def __init__(self, num_channels: int, fs: float, cutoff_hz: float, order: int = 1):
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff_hz / nyquist

        # Design SOS filter
        self._sos = butter(order, normalized_cutoff, btype="high", output="sos")

        # Precompute per-channel initial condition
        self._state = np.array([sosfilt_zi(self._sos) for _ in range(num_channels)])
        self._num_channels = num_channels

    def process(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        @brief Filter input data.

        @param data
            Array of shape [num_channels × num_samples].
        @return Filtered array (same shape).
        """
        num_channels, num_samples = data.shape

        out = np.zeros_like(data)

        for ch in range(num_channels):
            out[ch], self._state[ch] = sosfilt(
                self._sos,
                data[ch],
                zi=self._state[ch]
            )

        return out
