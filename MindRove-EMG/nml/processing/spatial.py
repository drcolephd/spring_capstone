# nml/processing/spatial.py
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray

class MontageMode(IntEnum):
    """
    @brief Enumeration of spatial referencing modes.

    @value MONOPOLAR
        Returns raw HPF EMG (no spatial subtraction).
    @value SINGLE_DIFFERENTIAL
        Subtracts immediate neighbor (circular array).
    @value LAPLACIAN
        Discrete Laplacian: subtracts average of two neighbors.
    """
    MONOPOLAR = 0
    SINGLE_DIFFERENTIAL = 1
    LAPLACIAN = 2

class SpatialReference:
    """
    @brief Spatial referencing (monopolar, SD, Laplacian) for 8-channel EMG.

    @param montage_mode
        Enum or int:
          0 = MONOPOLAR
          1 = SINGLE_DIFFERENTIAL
          2 = LAPLACIAN
    @type montage_mode: int
    """
    montage_mode: MontageMode = MontageMode.MONOPOLAR
    _montage: list[list[int]] = []

    def __init__(self, montage_mode: MontageMode | int = MontageMode.MONOPOLAR):
        # Accept either raw int or enum
        self.montage_mode = MontageMode(montage_mode)
        self._montage = self._make_montage(self.montage_mode)

    def _make_montage(self, mode: MontageMode):
        """
        @brief Generate adjacency list for spatial referencing.

        @param mode
            Montage mode enum.
        @type mode: MontageMode
        """
        if mode == MontageMode.LAPLACIAN:
            return [
                [1, 7],
                [0, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
                [5, 7],
                [6, 0],
            ]

        elif mode == MontageMode.SINGLE_DIFFERENTIAL:
            return [[1], [2], [3], [4], [5], [6], [7], [0]]

        else:  # MONOPOLAR
            return [[] for _ in range(8)]

    def set_mode(self, mode: MontageMode | int):
        """
        @brief Change montage mode and regenerate adjacency list.

        @param mode
            MontageMode enum or integer equivalent.
        """
        self.montage_mode = MontageMode(mode)
        self._montage = self._make_montage(self.montage_mode)

    def process(self, filtered_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        @brief Apply spatial referencing to HPF EMG.

        @param filtered_data
            Shape [8 × N].
        @return
            Spatially referenced EMG with same shape.
        @rtype NDArray[np.float64]
        """
        C, N = filtered_data.shape
        out = np.zeros_like(filtered_data)

        if self.montage_mode != MontageMode.MONOPOLAR:
            for ch in range(C):
                neighbors = self._montage[ch]
                neigh_mean = np.mean([filtered_data[n] for n in neighbors], axis=0)
                out[ch] = filtered_data[ch] - neigh_mean
        else:
            out[:, :] = filtered_data[:, :]

        return out
    