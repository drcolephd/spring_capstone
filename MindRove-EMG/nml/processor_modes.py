from __future__ import annotations
from enum import IntEnum


class ProcessorMode(IntEnum):
    OFF = 0
    ANGULAR_VELOCITY = 1          # old: _decode_mode == 1
    ANGULAR_ACCELERATION = 2      # old: _decode_mode == 2
    ANGULAR_VELOCITY_UNBOUNDED = 3  # old: _decode_mode == 3
    ANGULAR_VELOCITY_ROBOT = 4    # old: _decode_mode == 4
    GESTURE_CLASSIFICATION = 5

    @property
    def label(self) -> str:
        """Human-readable description for UI / logs."""
        if self is ProcessorMode.OFF:
            return "Off"
        if self is ProcessorMode.ANGULAR_VELOCITY:
            return "Angular velocity (bounded)"
        if self is ProcessorMode.ANGULAR_ACCELERATION:
            return "Angular acceleration (bounded)"
        if self is ProcessorMode.ANGULAR_VELOCITY_UNBOUNDED:
            return "Angular velocity (unbounded)"
        if self is ProcessorMode.ANGULAR_VELOCITY_ROBOT:
            return "Angular velocity (robot / LSL)"
        if self is ProcessorMode.GESTURE_CLASSIFICATION:
            return "Gesture classification (MLP)"
        return f"Unknown mode {int(self)}"

    def __str__(self) -> str:
        # Shows both descriptive label and enum name/value
        return f"{self.label} [{self.name}={int(self)}]"

    def __repr__(self) -> str:
        # Useful in debugger / REPL
        return f"ProcessorMode.{self.name}({int(self)})"
