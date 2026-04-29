import numpy as np
from numpy.typing import NDArray


class ImuArModel:
    """
    Per-channel online AR(p) model for IMU gyro data.

    Refits every refit_interval samples via OLS with Tikhonov regularisation.
    Produces 1-step prediction residuals (actual − predicted) via .residual.

    Defaults (500 Hz): order=8, fit_window=500 (≈1 s), refit_interval=125 (≈250 ms).
    """

    def __init__(
        self,
        order: int = 8,
        channels: int = 3,
        fit_window: int = 500,
        refit_interval: int = 125,
    ):
        self.order = order
        self.channels = channels
        self.fit_window = fit_window
        self.refit_interval = refit_interval

        self._buf = np.zeros((channels, fit_window), dtype=np.float64)
        self._coeffs = np.zeros((channels, order), dtype=np.float64)
        self._residual = np.zeros(channels, dtype=np.float32)
        self._write_pos = 0
        self._filled = 0
        self._since_refit = 0
        self.is_ready: bool = False

    def update(self, frame: NDArray) -> None:
        slot = self._write_pos % self.fit_window
        self._buf[:, slot] = frame[:self.channels]

        if self.is_ready and self._filled >= self.order:
            lags = [(self._write_pos - lag) % self.fit_window
                    for lag in range(1, self.order + 1)]
            lag_vals = self._buf[:, lags]               # (channels, order)
            pred = (self._coeffs * lag_vals).sum(axis=1)
            self._residual = (self._buf[:, slot] - pred).astype(np.float32)

        self._write_pos += 1
        if self._filled < self.fit_window:
            self._filled += 1
        self._since_refit += 1

        if self._filled >= self.fit_window and self._since_refit >= self.refit_interval:
            self._refit()
            self._since_refit = 0

    @property
    def residual(self) -> "NDArray[np.float32]":
        return self._residual.copy()

    def _refit(self) -> None:
        n_rows = self.fit_window - self.order
        if n_rows <= 0:
            return

        start = self._write_pos % self.fit_window
        t_slots = np.array(
            [(start + self.order + row) % self.fit_window for row in range(n_rows)]
        )

        for ch in range(self.channels):
            b = self._buf[ch]
            y = b[t_slots]
            X = np.column_stack(
                [b[(t_slots - 1 - lag) % self.fit_window] for lag in range(self.order)]
            )
            XtX = X.T @ X
            Xty = X.T @ y
            lam = (np.trace(XtX) / self.order) * 1e-3
            XtX += np.eye(self.order) * lam
            try:
                self._coeffs[ch] = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                pass

        self.is_ready = True

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._coeffs.fill(0.0)
        self._residual.fill(0.0)
        self._write_pos = 0
        self._filled = 0
        self._since_refit = 0
        self.is_ready = False


class ImuArFeatureExtractor:
    """
    Sliding window of ImuArModel residuals → 2 features per channel (RMS, variance).

    Default: 6 features for 3 channels, 100-sample window (≈200 ms @ 500 Hz).
    Call push() each IMU sample; call extract() at the MLP stride rate.
    """

    def __init__(self, channels: int = 3, window_samples: int = 100):
        self.channels = channels
        self.window_samples = window_samples
        self.feature_dim = channels * 2
        self._buf = np.zeros((channels, window_samples), dtype=np.float32)
        self._write_pos = 0
        self._filled = 0

    def push(self, residuals: "NDArray[np.float32]") -> None:
        slot = self._write_pos % self.window_samples
        self._buf[:, slot] = residuals[:self.channels]
        self._write_pos += 1
        if self._filled < self.window_samples:
            self._filled += 1

    def extract(self) -> "NDArray[np.float32]":
        result = np.zeros(self.feature_dim, dtype=np.float32)
        if self._filled < self.window_samples:
            return result

        start = self._write_pos % self.window_samples
        idx = (start + np.arange(self.window_samples)) % self.window_samples
        ordered = self._buf[:, idx].astype(np.float64)

        for ch in range(self.channels):
            x = ordered[ch]
            result[ch] = float(np.sqrt(np.mean(x * x)))
            result[self.channels + ch] = float(np.var(x))
        return result

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._write_pos = 0
        self._filled = 0
