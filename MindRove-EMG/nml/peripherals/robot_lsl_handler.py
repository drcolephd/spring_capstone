

import time
from nml.stream_manager import StreamManager, GestureCode

class RobotLSLHandler:
    """
    Encapsulates robot control via LSL / StreamManager.
    Keeps internal orientation & grasping state.
    """
    def __init__(self, stream: StreamManager) -> None:
        self.stream = stream
        self._orientation: float | None = None
        self._grasping: bool = False

    def home(self) -> None:
        """
        Run the hard-coded home sequence using GestureCode via StreamManager.
        """
        # Three groups of three CCW steps
        for _ in range(3):
            for _ in range(3):
                self.stream.send_single(GestureCode.WRIST_COUNTERCLOCKWISE)
                time.sleep(0.25)

        # Three RELEASE pulses
        for _ in range(3):
            self.stream.send_single(GestureCode.RELEASE)
            time.sleep(0.5)

    def update(
        self,
        dx: float,
        dy: float,
        power: float,
        orientation: float,
        assertion_threshold: float,
        power_threshold: float,
        orientation_threshold: float,
        enabled: bool,
        state_change_cb=None,
    ) -> None:
        """
        Main robot update logic (formerly update_lsl_robot_message).

        state_change_cb: callable(bool) or None, used to emit state changes.
        """
        if self._orientation is None:
            self._orientation = orientation

        delta_orientation = orientation - self._orientation

        if not enabled:
            return

        # Orientation-based sup/pro
        if abs(delta_orientation) > orientation_threshold:
            self._orientation = orientation
            if delta_orientation > 0:
                self.stream.send_single(GestureCode.WRIST_SUP)
            else:
                self.stream.send_single(GestureCode.WRIST_PRO)

        # dx, dy -> wrist rotation / flexion/extension
        if dx > assertion_threshold:
            self.stream.send_single(GestureCode.WRIST_CLOCKWISE)
        elif dx < -assertion_threshold:
            self.stream.send_single(GestureCode.WRIST_COUNTERCLOCKWISE)

        if dy > assertion_threshold:
            self.stream.send_single(GestureCode.WRIST_EXTEND)
        elif dy < -assertion_threshold:
            self.stream.send_single(GestureCode.WRIST_FLEX)

        # Power-based grasp / release
        if self._grasping:
            if power < power_threshold:
                self.stream.send_single(GestureCode.RELEASE)
                if state_change_cb is not None:
                    state_change_cb(False)
                self._grasping = False
        else:
            if power > power_threshold:
                self.stream.send_single(GestureCode.GRASP)
                if state_change_cb is not None:
                    state_change_cb(True)
                self._grasping = True