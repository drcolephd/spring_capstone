import numpy as np
from numpy.typing import NDArray


def normalize_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quaternion_conjugate(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_inverse(q: NDArray[np.float64]) -> NDArray[np.float64]:
    qc = quaternion_conjugate(q)
    d = float(np.dot(q, q))
    if d <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return qc / d


def quaternion_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    qn = normalize_quaternion(q)
    w, x, y, z = qn
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_xyz_rad_to_quaternion(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return normalize_quaternion(q)


class ArmIncrementalKinematics:
    """
    Dual-IMU arm incremental kinematics (paper Eq. 2-4 + filtered updates Eq. 12-17).

    Coordinate assumptions:
    - Upper arm quaternion is R^S_E (elbow frame w.r.t shoulder frame).
    - Forearm quaternion is R^S_W (wrist frame w.r.t shoulder frame).
    - Segment vectors are aligned with local +X axes.
    """

    def __init__(self, upper_arm_length_m: float, forearm_length_m: float, smooth_window: int = 10):
        self.upper_arm_length_m = upper_arm_length_m
        self.forearm_length_m = forearm_length_m
        self.smooth_window = max(1, int(smooth_window))

        self._prev_wrist_pos = np.zeros(3, dtype=np.float64)
        self._prev_forearm_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._initialized = False

        self._dp_hist = np.zeros((self.smooth_window, 3), dtype=np.float64)
        self._dq_hist = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (self.smooth_window, 1))
        self._hist_count = 0
        self._hist_write = 0

    def _average_quaternion(self) -> NDArray[np.float64]:
        if self._hist_count <= 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        m = self._dq_hist[: self._hist_count].T  # shape 4 x N
        u, _, _ = np.linalg.svd(m, full_matrices=False)
        return normalize_quaternion(u[:, 0].astype(np.float64))

    def update(
        self,
        upper_q: NDArray[np.float64],
        forearm_q: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        upper_q = normalize_quaternion(upper_q.astype(np.float64))
        forearm_q = normalize_quaternion(forearm_q.astype(np.float64))

        r_se = quaternion_to_rotation_matrix(upper_q)
        r_sw = quaternion_to_rotation_matrix(forearm_q)

        p_se_e = np.array([self.upper_arm_length_m, 0.0, 0.0], dtype=np.float64)
        p_ew_w = np.array([self.forearm_length_m, 0.0, 0.0], dtype=np.float64)
        wrist_pos = r_se @ p_se_e + r_sw @ p_ew_w

        if not self._initialized:
            self._prev_wrist_pos = wrist_pos.copy()
            self._prev_forearm_q = forearm_q.copy()
            self._initialized = True
            return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        delta_p = wrist_pos - self._prev_wrist_pos
        delta_q = quaternion_multiply(forearm_q, quaternion_inverse(self._prev_forearm_q))
        delta_q = normalize_quaternion(delta_q)

        self._prev_wrist_pos = wrist_pos.copy()
        self._prev_forearm_q = forearm_q.copy()

        i = self._hist_write % self.smooth_window
        self._dp_hist[i] = delta_p
        self._dq_hist[i] = delta_q
        self._hist_write += 1
        if self._hist_count < self.smooth_window:
            self._hist_count += 1

        dps = self._dp_hist[: self._hist_count]
        delta_p_avg = np.mean(dps, axis=0)
        delta_q_avg = self._average_quaternion()
        return delta_p, delta_q, delta_p_avg, delta_q_avg


def wrist_position_decomposition(
    upper_q: NDArray[np.float64],
    forearm_q: NDArray[np.float64],
    upper_arm_length_m: float,
    forearm_length_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Shoulder-frame wrist position split into upper-arm vs forearm contributions (paper Eq. 2).

    Returns (p_from_upper, p_from_forearm, p_total) each in R^3.
    """
    upper_q = normalize_quaternion(upper_q.astype(np.float64))
    forearm_q = normalize_quaternion(forearm_q.astype(np.float64))
    r_se = quaternion_to_rotation_matrix(upper_q)
    r_sw = quaternion_to_rotation_matrix(forearm_q)
    p_se_e = np.array([upper_arm_length_m, 0.0, 0.0], dtype=np.float64)
    p_ew_w = np.array([forearm_length_m, 0.0, 0.0], dtype=np.float64)
    p_upper = r_se @ p_se_e
    p_fore = r_sw @ p_ew_w
    return p_upper, p_fore, p_upper + p_fore
