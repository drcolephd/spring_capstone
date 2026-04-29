# nml/processing/activity.py
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi


class MotorTwitchHandler:
    """
    @brief Extract twitch-like motor activity from accelerometer signals.

    @details
    Pipeline:
      1. Compute gravity vector as mean(acc)
      2. Subtract gravity → yields vibration-only signal
      3. RMS across axes
      4. High-pass filter @ 50 Hz
      5. Output twitch magnitude

    @param fs
        Sampling frequency (Hz).
    @type fs: float
    """

    def __init__(self, fs: float):
        self.fs = fs

        nyq = 0.5 * fs
        cutoff = 50.0 / nyq  # HPF for twitch detection

        self._sos = butter(2, cutoff, btype="high", output="sos")
        self._state = sosfilt_zi(self._sos)

        # Running gravity estimate (simple mean)
        self._gravity = np.zeros(3, dtype=np.float64)

    def process(self, acc_samples: NDArray[np.float64]) -> float:
        """
        @brief Compute twitch magnitude.

        @param acc_samples
            Accelerometer array [3 × N].
        @return
            High-pass filtered RMS vibration amplitude.
        @rtype float
        """
        # Mean gravity vector
        self._gravity = np.mean(acc_samples, axis=1)

        # Subtract gravity from each sample
        acc_no_g = acc_samples - self._gravity[:, None]  # [3 × N]

        # Channel-wise RMS → 1D vibration signal
        rms_signal = np.sqrt(np.mean(acc_no_g**2, axis=0))  # shape [N]

        # HPF filter to isolate twitch-like bursts
        twitch, self._state = sosfilt(self._sos, rms_signal, zi=self._state)

        # Return last sample (current time)
        return float(twitch[-1])
