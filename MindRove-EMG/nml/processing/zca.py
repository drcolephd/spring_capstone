# nml/zca.py
import numpy as np
from numpy.typing import NDArray

class ZcaParams:
    """
    @brief Parameter container for ZCA whitening.

    @param num_channels 
        The number of EMG channels to whiten.
    @type num_channels: int

    @param buffer_duration_sec
        Duration of the buffer (in seconds) collected prior to computing
        the whitening transform.
    @type buffer_duration_sec: float

    @param tikhonov_epsilon
        Tikhonov regularization ε added to eigenvalues before inversion.
    @type tikhonov_epsilon: float

    @param enable
        Whether ZCA whitening is enabled.
    @type enable: bool
    """

    num_channels: int
    buffer_duration_sec: float
    tikhonov_epsilon: float
    enable: bool

    def __init__(
        self,
        num_channels: int = 8,
        buffer_duration_sec: float = 5.0,
        tikhonov_epsilon: float = 1.5,
        enable: bool = True,
    ):
        self.num_channels = num_channels
        self.buffer_duration_sec = buffer_duration_sec
        self.tikhonov_epsilon = tikhonov_epsilon
        self.enable = enable
        self._validate()

    # ---------------------------------------------------------
    # Validation helper
    # ---------------------------------------------------------
    def _validate(self):
        """
        @brief Validate parameter values.
        """
        if self.num_channels <= 0:
            raise ValueError("ZcaParams.num_channels must be > 0.")
        if self.buffer_duration_sec <= 0:
            raise ValueError("ZcaParams.buffer_duration_sec must be > 0.")
        if self.tikhonov_epsilon <= 0:
            raise ValueError("ZcaParams.tikhonov_epsilon must be > 0.")

    # ---------------------------------------------------------
    # Dunder methods
    # ---------------------------------------------------------
    def __repr__(self) -> str:
        """
        @brief Official string representation (debug-friendly).
        @return Representation string.
        @rtype str
        """
        return (
            f"ZcaParams(num_channels={self.num_channels}, "
            f"buffer_duration_sec={self.buffer_duration_sec}, "
            f"tikhonov_epsilon={self.tikhonov_epsilon}, "
            f"enable={self.enable})"
        )

    def __str__(self) -> str:
        """
        @brief User-friendly string representation.
        @return Readable summary string.
        @rtype str
        """
        return (
            "ZCA Whiten Parameters:\n"
            f"  Channels           : {self.num_channels}\n"
            f"  Buffer Duration    : {self.buffer_duration_sec:.3f} s\n"
            f"  Tikhonov Epsilon   : {self.tikhonov_epsilon}\n"
            f"  Enabled            : {self.enable}"
        )

    def __eq__(self, other) -> bool:
        """
        @brief Structural equality check.
        @return True if fields match, otherwise False.
        @rtype bool
        """
        if not isinstance(other, ZcaParams):
            return False
        return (
            self.num_channels == other.num_channels and
            self.buffer_duration_sec == other.buffer_duration_sec and
            self.tikhonov_epsilon == other.tikhonov_epsilon and
            self.enable == other.enable
        )

    # ---------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------
    def copy(self):
        """
        @brief Create a copy of the parameter object.
        @return New ZcaParams instance.
        @rtype ZcaParams
        """
        return ZcaParams(
            num_channels=self.num_channels,
            buffer_duration_sec=self.buffer_duration_sec,
            tikhonov_epsilon=self.tikhonov_epsilon,
            enable=self.enable,
        )

    @classmethod
    def from_dict(cls, d: dict):
        """
        @brief Create a ZcaParams object from a dictionary.

        @param d
            Dictionary with any subset of keys:
            num_channels, buffer_duration_sec, tikhonov_epsilon, enable.
        @type d: dict

        @return A new ZcaParams instance.
        @rtype ZcaParams
        """
        return cls(
            num_channels=d.get("num_channels", 8),
            buffer_duration_sec=d.get("buffer_duration_sec", 5.0),
            tikhonov_epsilon=d.get("tikhonov_epsilon", 1.5),
            enable=d.get("enable", True),
        )


class ZcaHandler:
    """
    @brief Online single-shot ZCA whitening for multi-channel EMG.

    @details
    The ZCA object:
      - accumulates data until `num_buffer_samples` is reached
      - computes channel-wise mean, global sigma, and ZCA matrix using a correlation matrix
      - applies whitened projection to subsequent samples

    The whitening rule is:

        Z = ((X - mu)/sigma) @ W

    Where:
      - `mu` is channel-wise mean
      - `sigma` is global standard deviation
      - `W` is the ZCA whitening matrix: E * diag(1/sqrt(D + eps)) * E'
    """

    _buffer_count: int = 0
    _buffer: NDArray[np.float64]

    _num_channels: int = 0
    _num_buffer_samples: int = 0
    _fs: float = 0.0
    _eps: float = 0.0
    _enabled: bool = False
    _trained: bool = False

    _mu: NDArray[np.float64] | None = None
    _sigma: float | None = None
    _W: NDArray[np.float64] | None = None

    def __init__(self, fs: float, params: ZcaParams):
        """
        @brief Construct a ZCA whitening object.

        @param fs 
            Sampling frequency in Hz.
        @type fs: float

        @param params
            A ZcaParams instance defining the whitening configuration.
        @type params: ZcaParams
        """
        self._num_channels = params.num_channels
        self._num_buffer_samples = int(params.buffer_duration_sec * fs)
        self._fs = fs
        self._eps = params.tikhonov_epsilon
        self._enabled = params.enable

        self._buffer = np.zeros(
            (self._num_channels, self._num_buffer_samples), dtype=np.float64
        )

    @property
    def trained(self) -> bool:
        """
        @brief Indicates whether the whitening matrix has been computed.

        @return True if whitening has been computed, otherwise False.
        @rtype bool
        """
        return self._trained
    
    @property
    def full(self) -> bool:
        """
        @brief Indicates whether the whitening buffer is full.

        @return True if the buffer is full, otherwise False.
        @rtype bool
        """
        return self._trained

    def update_buffer(self, data: NDArray[np.float64]):
        """
        @brief Append samples to the whitening buffer until full.

        @param data
            Incoming EMG data with shape [num_channels × num_samples].
        @type data: NDArray[np.float64]

        @note
            Once the buffer fills, `_compute_whitener()` is automatically called.
        """
        if self._trained or not self._enabled:
            return

        ns = data.shape[1]
        remaining = self._num_buffer_samples - self._buffer_count
        take = min(ns, remaining)

        self._buffer[:, self._buffer_count : self._buffer_count + take] = data[:, :take]
        self._buffer_count += take

        if self._buffer_count >= self._num_buffer_samples:
            self._compute_whitener()

    def _compute_whitener(self):
        """
        @brief Compute mean, sigma, and the ZCA whitening matrix.

        @details
        Uses correlation-matrix ZCA:

            C = corrcoef(Xn)
            [E, D] = eig(C)
            W = E * diag(1/sqrt(D + eps)) * E'

        @note
            This operation is performed only once. After computation,
            the internal buffer is deleted to free memory.
        """
        X = self._buffer.T

        mu = X.mean(axis=0)
        sigma = X.std()

        Xn = (X - mu) / sigma

        C = np.corrcoef(Xn, rowvar=False)

        D, E = np.linalg.eig(C)

        D_inv_sqrt = np.diag(1.0 / np.sqrt(D + self._eps))
        W = E @ D_inv_sqrt @ E.T

        self._mu = mu
        self._sigma = sigma
        self._W = W
        self._trained = True

        del self._buffer

    def apply(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        @brief Apply whitening to data.

        @param data
            EMG array of shape [num_channels × num_samples].
        @type data: NDArray[np.float64]

        @return Whitened EMG if trained; otherwise returns input unchanged.
        @rtype NDArray[np.float64]

        @note 
            Uses matrix multiplication on sample-major data for efficiency.
        """
        if (not self._trained) or (not self._enabled):
            return data

        X = data.T
        Xn = (X - self._mu) / self._sigma
        Z = Xn @ self._W

        return Z.T

    def enable(self):
        """
        @brief Enable ZCA whitening.
        """
        self._enabled = True

    def disable(self):
        """
        @brief Disable ZCA whitening.
        """
        self._enabled = False

    def reset(
        self,
        new_num_channels: int | None = None,
        new_buffer_duration_sec: float | None = None,
    ):
        """
        @brief Reset the whitening state and optionally resize parameters.

        @param new_num_channels
            Optional new number of channels.
        @type new_num_channels: int | None

        @param new_buffer_duration_sec
            Optional new buffer duration in seconds.
        @type new_buffer_duration_sec: float | None

        @note 
            After reset, whitening must re-train from scratch.
        """
        if new_num_channels is not None:
            self._num_channels = new_num_channels
        if new_buffer_duration_sec is not None:
            self._num_buffer_samples = int(new_buffer_duration_sec * self._fs)

        self._trained = False
        self._buffer = np.zeros(
            (self._num_channels, self._num_buffer_samples), dtype=np.float64
        )
        self._buffer_count = 0
