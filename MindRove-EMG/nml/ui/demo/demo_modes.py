from __future__ import annotations
from enum import Enum

class DemoMode(Enum):
    VIGEM_LATCH = "ViGEM Latching Button Decoder"
    LSL_STRETCH = "Stretch-Robot LSL Controller"
    DECODE_KB = "Keyboard MC Decoder"
    DECODE_MOUSE = "2D Wrist-Mouse Decoder"

    def __str__(self) -> str:
        return self.value
