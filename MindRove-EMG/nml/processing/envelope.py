# nml/processing/envelope.py
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi


class EnvelopeSmoother:
    """
    @brief Envelope smoother for rectified EMG using a low-pass SOS filter.

    @param num_channels
        Number of EMG channels to smooth.
    @type num_channels: int

    @param fs
        Sampling frequency (Hz).
    @type fs: float

    @param cutoff_hz
        Low-pass cutoff frequency for envelope smoothing (Hz).
    @type cutoff_hz: float

    @param order
        Filter order.
    @type order: int
    """

    def __init__(self, num_channels: int, fs: float, cutoff_hz: float, order: int = 1):
        self.num_channels = num_channels
        self.fs = fs

        nyq = 0.5 * fs
        norm_cut = cutoff_hz / nyq

        self._sos = butter(order, norm_cut, btype="low", output="sos")
        self._state = np.array([sosfilt_zi(self._sos) for _ in range(num_channels)])

    def process(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        @brief Rectify EMG and apply low-pass envelope smoothing.

        @param data
            Shape [num_channels × num_samples].
        @return
            Envelope data, shape [num_channels × num_samples].
        """
        C, N = data.shape
        out = np.zeros((C, N), dtype=np.float64)

        # Full-wave rectification → low-pass filter
        for ch in range(C):
            rect = np.abs(data[ch])
            out[ch], self._state[ch] = sosfilt(
                self._sos, rect, zi=self._state[ch]
            )

        return out
