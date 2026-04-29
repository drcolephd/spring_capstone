from nml.vigem import ViGEmPy, ViGEmEnum

class GamepadController:
    """
    Thin wrapper around ViGEmPy that can be initialized/cleaned up on demand.
    """
    def __init__(self) -> None:
        self._pad: ViGEmPy | None = None

    def initialize(self) -> None:
        """Initialize underlying ViGEm gamepad if not already initialized."""
        if self._pad is None:
            pad = ViGEmPy()
            pad.initialize()
            self._pad = pad

    def cleanup(self) -> None:
        """Clean up the underlying ViGEm gamepad."""
        if self._pad is not None:
            try:
                self._pad.cleanup()
            except Exception:
                print("[GamepadController]::Did not clean up the ViGeM gamepad.")
            finally:
                self._pad = None

    @property
    def is_ready(self) -> bool:
        return self._pad is not None

    @property
    def buttons(self) -> int:
        if self._pad is None:
            return 0x0000
        return self._pad.buttons

    def send_input(self, btn: int) -> None:
        if self._pad is not None:
            self._pad.send_input(btn)
