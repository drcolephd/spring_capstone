# nml/processing/gyros.py
import numpy as np
from numpy.typing import NDArray


class GyroProcessor:
    """
    @brief Lightweight wrapper for processing gyroscope samples.

    @param scalar
        Scaling factor applied to raw gyro samples.
    @type scalar: float
    """

    def __init__(self, scalar: float):
        self.scalar = scalar

    def process(self, gyro_samples: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        @brief Apply scalar and integrate gyro axes.

        @param gyro_samples
            Raw gyro [3 × N] array.
        @return
            Integrated gyro contribution (shape [3]).
        """
        # Sum over time window (integrate) after scaling
        return np.sum(gyro_samples * self.scalar, axis=1)
