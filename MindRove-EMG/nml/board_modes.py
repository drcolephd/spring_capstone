from __future__ import annotations

from enum import IntEnum

class BoardMode(IntEnum):
    NONE = -1
    DPAD_A_LATCH = 3
    DPAD_A_CONTINUOUS = 4
    DIRECTED_A_LATCH = 5
    NONDIRECTED_A_LATCH_TOGGLE = 6
    DPAD_WINNER_A_CONTINUOUS = 7
    WRIST_LSL_CONTINUOUS = 8
    WRIST_2D_DECODE_CONTINUOUS = 9
    DIGIT_4BUTTON_DECODE_DISCRETE = 10

    @property
    def label(self) -> str:
        if self is BoardMode.NONE:
            return "Off"
        if self is BoardMode.DPAD_A_LATCH:
            return "DPad + A (Press/Latch)"
        if self is BoardMode.DPAD_A_CONTINUOUS:
            return "DPad + A (Press/Continuous)"
        if self is BoardMode.DIRECTED_A_LATCH:
            return "Directed A-Only (Latch)"
        if self is BoardMode.NONDIRECTED_A_LATCH_TOGGLE:
            return "Non-Directed A-Only (Latch-Toggle)"
        if self is BoardMode.DPAD_WINNER_A_CONTINUOUS:
            return "DPad-WINNER + A (Press/Continuous)"
        if self is BoardMode.WRIST_LSL_CONTINUOUS:
            return "Wrist LSL (Continuous)"
        if self is BoardMode.WRIST_2D_DECODE_CONTINUOUS:
            return "Wrist 2-Axis Cursor (Continuous)"
        if self is BoardMode.DIGIT_4BUTTON_DECODE_DISCRETE:
            return "Digit 4-Button Classifier (MC, Discrete)"
        # Optional fallback
        return f"Mode {int(self)}"
