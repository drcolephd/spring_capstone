# nml/processing/orientation.py
import numpy as np
from numpy.typing import NDArray


class OrientationEstimator:
    """
    @brief Complementary-filter orientation estimator.

    @param alpha
        Gyro gain (0–1). Accelerometer gain = (1 - alpha).
    @type alpha: float

    @param wrist_gain
        Optional scaling applied to wrist orientation estimation.
    @type wrist_gain: float
    """

    def __init__(self, alpha: float = 0.995, wrist_gain: float = 1.0):
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.wrist_gain = wrist_gain

        self._orientation = np.zeros(3, dtype=np.float64)
        self._wrist_orientation = np.zeros(3, dtype=np.float64)

    def compute_acc_angles(self, acc_mean: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        @brief Compute pitch/roll angles from accelerometer.

        @param acc_mean
            Mean accelerometer vector [3].
        @return
            Two-axis gravity orientation estimate.
        """
        ax, ay, az = acc_mean
        out = np.zeros(3)
        out[0] = np.arctan2(ay, az)
        out[1] = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        return out

    def update(self, gyro_contrib: NDArray[np.float64], acc_angles: NDArray[np.float64]):
        """
        @brief Update orientation estimates.

        @param gyro_contrib
            Integrated gyro values [3].
        @param acc_angles
            Pitch/roll derived from accelerometer [3].
        """
        self._orientation = (
            self.alpha * (gyro_contrib + self._orientation)
            + self.beta * acc_angles
        )
        self._wrist_orientation = (
            self.alpha * (gyro_contrib * self.wrist_gain + self._wrist_orientation)
            + self.beta * (acc_angles * self.wrist_gain)
        )

    @property
    def orientation(self) -> NDArray[np.float64]:
        """@return Current orientation estimate."""
        return self._orientation

    @property
    def wrist_orientation(self) -> NDArray[np.float64]:
        """@return Wrist-adjusted orientation estimate."""
        return self._wrist_orientation
