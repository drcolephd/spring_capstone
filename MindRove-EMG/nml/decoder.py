from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nml.processor_modes import ProcessorMode

if TYPE_CHECKING:
    # Only for type hints; avoids runtime circular import
    from nml.processor import Processor


class Decoder:
    """
    Encapsulates the EMG → control decoding logic for Processor.

    It owns *no* buffers itself; it reads and writes state on the
    Processor instance it wraps and emits signals via Processor's
    PyQt signals.
    """

    def __init__(self, processor: "Processor") -> None:
        self._p = processor  # reference to owning Processor

    def step(self, env_data: np.ndarray) -> None:
        """
        Perform one decode/update step given the latest env_data.

        env_data: shape (9, num_samples) as produced by Processor.update.
        """
        p = self._p

        # ------------------------------------------------------------------
        # 1) Model-based decoding branch (rates model already learned)
        # ------------------------------------------------------------------
        if p._has_rates_model:
            # Use learned rates model: _rates_x (features) and _PLS_BETA
            p.stream_socket.send_rates(p._rates_x[1:], scale_factor=100)
            p._delta_omega = p._rates_x @ p._PLS_BETA

            omega_prev = np.copy(p._omega)
            p._omega[0] = min(
                max(
                    p._omega[0]
                    + p._delta_omega[0] * p._omega_gain[0],
                    -p._omega_limit,
                ),
                p._omega_limit,
            )
            p._omega[1] = min(
                max(
                    p._omega[1]
                    + p._delta_omega[1] * p._omega_gain[1],
                    -p._omega_limit,
                ),
                p._omega_limit,
            )

            delta_omega = p._omega - omega_prev
            global_env_power = np.mean(p._env_history)
            # Same rotated sign convention as before
            p.delta_omega.emit(
                -delta_omega[1],
                delta_omega[0],
                global_env_power,
                p.wrist_orientation[0],
            )
            return  # nothing else to do for model-based mode

        # ------------------------------------------------------------------
        # 2) Non-model decoding based on _decode_mode
        # ------------------------------------------------------------------
        if p._decode_mode is ProcessorMode.ANGULAR_VELOCITY:
            # angular velocity (bounded)
            p._delta_omega = env_data.T @ p._PLS_BETA

            p._omega[0] = min(
                max(
                    p._omega[0]
                    + np.sum(
                        np.where(
                            np.abs(p._delta_omega[:, 0]) > p._omega_threshold[0],
                            p._delta_omega[:, 0],
                            0,
                        )
                    )
                    * p._omega_gain[0],
                    -p._omega_limit,
                ),
                p._omega_limit,
            )
            p._omega[1] = min(
                max(
                    p._omega[1]
                    + np.sum(
                        np.where(
                            np.abs(p._delta_omega[:, 1]) > p._omega_threshold[1],
                            p._delta_omega[:, 1],
                            0,
                        )
                    )
                    * p._omega_gain[1],
                    -p._omega_limit,
                ),
                p._omega_limit,
            )

            p.omega.emit(
                p._omega,
                env_data,
                p._omega_threshold,
                p._omega_gain,
                p._has_rates_model,
            )

        elif p._decode_mode is ProcessorMode.ANGULAR_ACCELERATION:
            # angular acceleration (bounded)
            delta_delta_omega = env_data.T @ p._PLS_BETA

            p._delta_omega[0] += min(
                max(
                    np.sum(
                        np.where(
                            np.abs(delta_delta_omega[:, 0])
                            > p._omega_threshold[0] * 0.001,
                            delta_delta_omega[:, 0],
                            0,
                        )
                    ),
                    -0.1 * p._omega_limit,
                ),
                0.1 * p._omega_limit,
            )
            p._delta_omega[1] += min(
                max(
                    np.sum(
                        np.where(
                            np.abs(delta_delta_omega[:, 1])
                            > p._omega_threshold[1] * 0.001,
                            delta_delta_omega[:, 1],
                            0,
                        )
                    ),
                    -0.1 * p._omega_limit,
                ),
                0.1 * p._omega_limit,
            )

            p._omega[0] = min(
                max(
                    p._omega[0]
                    + np.sum(
                        np.where(
                            np.abs(p._delta_omega[0]) > p._omega_threshold[0],
                            p._delta_omega[0],
                            0,
                        )
                    )
                    * p._omega_gain[0],
                    -p._omega_limit,
                ),
                p._omega_limit,
            )
            p._omega[1] = min(
                max(
                    p._omega[1]
                    + np.sum(
                        np.where(
                            np.abs(p._delta_omega[1]) > p._omega_threshold[1],
                            p._delta_omega[1],
                            0,
                        )
                    )
                    * p._omega_gain[1],
                    -p._omega_limit,
                ),
                p._omega_limit,
            )

            p.omega.emit(
                p._omega,
                env_data,
                p._omega_threshold,
                p._omega_gain,
                p._has_rates_model,
            )

        elif p._decode_mode is ProcessorMode.ANGULAR_VELOCITY_UNBOUNDED:
            # angular velocity; no bounding
            p._delta_omega = env_data.T @ p._PLS_BETA

            delta_omega_x = np.sum(
                np.where(
                    np.abs(p._delta_omega[:, 0]) > p._omega_threshold[0],
                    p._delta_omega[:, 0],
                    0,
                )
            )
            delta_omega_y = np.sum(
                np.where(
                    np.abs(p._delta_omega[:, 1]) > p._omega_threshold[1],
                    p._delta_omega[:, 1],
                    0,
                )
            )

            p._omega[0] = p._omega[0] + delta_omega_x
            p._omega[1] = p._omega[1] + delta_omega_y

            global_env_power = np.mean(p._env_history)
            p.delta_omega.emit(
                delta_omega_x,
                delta_omega_y,
                global_env_power,
                p.wrist_orientation[0],
            )

        elif p._decode_mode is ProcessorMode.ANGULAR_VELOCITY_ROBOT:
            # angular velocity for robot/LSL, with bounding & rotated axes
            p._delta_omega = env_data.T @ p._PLS_BETA

            omega_prev = np.copy(p._omega)
            p._omega[0] = min(
                max(
                    p._omega[0]
                    + np.sum(
                        np.where(
                            np.abs(p._delta_omega[:, 0]) > p._omega_threshold[0],
                            p._delta_omega[:, 0],
                            0,
                        )
                    )
                    * p._omega_gain[0],
                    -p._omega_limit,
                ),
                p._omega_limit,
            )
            p._omega[1] = min(
                max(
                    p._omega[1]
                    + np.sum(
                        np.where(
                            np.abs(p._delta_omega[:, 1]) > p._omega_threshold[1],
                            p._delta_omega[:, 1],
                            0,
                        )
                    )
                    * p._omega_gain[1],
                    -p._omega_limit,
                ),
                p._omega_limit,
            )

            delta_omega = p._omega - omega_prev
            global_env_power = np.mean(p._env_history)
            p.delta_omega.emit(
                -delta_omega[1],
                delta_omega[0],
                global_env_power,
                p.wrist_orientation[0],
            )

        # OFF or unrecognized modes: no-op (could log if desired)
