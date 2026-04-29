import logging
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QTimer, QObject
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams
from nml.processor import Processor
from nml.peripherals.gamepad_controller import GamepadController
from nml.peripherals.robot_lsl_handler import RobotLSLHandler
from nml.vigem import ViGEmEnum
from nml.board_modes import BoardMode
from nml.binary_logger import BinaryLogger
from nml.binary_reader import BinaryReader
from nml.stream_manager import StreamManager
from nml.ui.demo.demo_modes import DemoMode
from nml.ui.demo.board_manager_window import BoardManagerWindow


class BoardManager(QObject):
    x: float = 0.0
    y: float = 0.0

    asserted: bool = False
    _enabled: bool = True

    stream_started = pyqtSignal()
    stream_stopped = pyqtSignal(int)
    filename_updated = pyqtSignal(str, int)
    state_change = pyqtSignal(bool)  # Asserted: True | Deasserted: False

    buffer_size: int = 256
    board_shim: BoardShim | None = None
    processor: Processor | None = None

    # Controllers / external interfaces
    gamepad: GamepadController | None = None
    _lsl_manager: StreamManager | None = None
    _robot: RobotLSLHandler | None = None

    _running = False
    _in_debounce = False
    _assertion_debounce_ms = 1500
    _deassertion_debounce_ms = 1500
    _debounce_timer: QTimer | None = None

    _save_logs = False
    filename: str | None = None
    suffix: int | None = None
    logger: BinaryLogger | None = None

    _a_button_bytecode = ViGEmEnum.encode(["A"])
    _up_button_bytecode = ViGEmEnum.encode(["D-Pad Up"])
    _down_button_bytecode = ViGEmEnum.encode(["D-Pad Down"])
    _left_button_bytecode = ViGEmEnum.encode(["D-Pad Left"])
    _right_button_bytecode = ViGEmEnum.encode(["D-Pad Right"])

    _assertion_threshold: float = 0.0016
    _deassertion_threshold: float = 0.04
    _power_threshold: float = 5000000
    _orientation_threshold: float = 0.01

    # 3–7: gamepad modes, 8: robot/LSL
    _mode: BoardMode = BoardMode.NONE

    # App type controls which peripherals we spin up
    _app_type: DemoMode | None = None  

    return_direction: str = "NONE"

    _viewer: BoardManagerWindow | None = None

    def __init__(
        self,
        synth: int = 0,
        app_type: DemoMode = DemoMode.VIGEM_LATCH,
        filename: str | None = None,
        suffix: int | None = None,
        buffer_size: int | None = None,
        save_logs: bool = False,
        compute_thresholds: bool = False,
        do_spike_detection: bool = False,
    ):
        """Initialize the BoardManager object."""
        super().__init__()

        # ------------------------------
        # App type configuration
        # ------------------------------
        self._app_type = app_type

        # Map app_type -> initial mode
        # (you can tweak these if you want different behaviors)
        if self._app_type == DemoMode.VIGEM_LATCH:
            self._mode = BoardMode.NONDIRECTED_A_LATCH_TOGGLE
        elif self._app_type == DemoMode.LSL_STRETCH:
            self._mode = BoardMode.WRIST_LSL_CONTINUOUS
        elif self._app_type == DemoMode.DECODE_KB:
            self._mode = BoardMode.DIGIT_4BUTTON_DECODE_DISCRETE
        elif self._app_type == DemoMode.DECODE_MOUSE:
            self._mode = BoardMode.WRIST_2D_DECODE_CONTINUOUS

        # ------------------------------
        # Handle optional inputs
        # ------------------------------
        if synth == 0:
            self.board_id = BoardIds.MINDROVE_WIFI_BOARD
        else:
            self.board_id = BoardIds.SYNTHETIC_BOARD

        if buffer_size is not None:
            self.buffer_size = buffer_size
        if filename is not None:
            self.filename = filename
        if save_logs is not None:
            self._save_logs = save_logs
        if suffix is not None:
            self.suffix = suffix

        # ------------------------------
        # LSL / robot stream manager & handler
        # Only for STRETCH / CAL1 variants
        # ------------------------------
        if self._app_type in (DemoMode.LSL_STRETCH, ):
            self._lsl_manager = StreamManager()
            self._robot = RobotLSLHandler(self._lsl_manager)
        else:
            self._lsl_manager = None
            self._robot = None

        # ------------------------------
        # Logging
        # ------------------------------
        BoardShim.enable_dev_board_logger()
        logging.basicConfig(level=logging.DEBUG)

        # ------------------------------
        # MindRove board
        # ------------------------------
        params = MindRoveInputParams()
        self.board_shim = BoardShim(self.board_id, params)
        s = BoardShim.get_board_descr(self.board_id)
        print(s)

        # Prepare the board session
        self.board_shim.prepare_session()

        # ------------------------------
        # Gamepad (only if mode / app type uses it)
        # ------------------------------
        if self._mode in (
            BoardMode.DPAD_A_LATCH,
            BoardMode.DPAD_A_CONTINUOUS,
            BoardMode.DIRECTED_A_LATCH,
            BoardMode.NONDIRECTED_A_LATCH_TOGGLE,
            BoardMode.DPAD_WINNER_A_CONTINUOUS,
        ):
            self.gamepad = GamepadController()
            self.gamepad.initialize()
        else:
            self.gamepad = None

        # ------------------------------
        # Signal processor
        # ------------------------------
        self.processor = Processor(
            self.board_shim,
            self.buffer_size,
            compute_thresholds=compute_thresholds,
            do_spike_detection=do_spike_detection,
        )

        # Initial wiring based on mode
        self._wire_processor_for_mode(self._mode.value)

        # ------------------------------
        # Debounce timer
        # ------------------------------
        self._debounce_timer = QTimer()
        self._debounce_timer.timeout.connect(self.debounce_timeout_handler)

    def __del__(self):
        """Clean up the ViGeM gamepad when the object is deleted."""
        try:
            if self.gamepad is not None:
                self.gamepad.cleanup()
        except Exception:
            print("[BoardManager]::Did not clean up the ViGeM gamepad.")

    # ------------------------------------------------------------------
    # UI / Window
    # ------------------------------------------------------------------
    def open(self):
        """Opens the BoardManagerWindow instance."""
        if self._viewer is None:
            # Decide which modes are allowed in the UI for this app_type
            if self._app_type == DemoMode.VIGEM_LATCH:
                allowed_modes = [
                    BoardMode.NONDIRECTED_A_LATCH_TOGGLE,
                    BoardMode.DIRECTED_A_LATCH,
                ]
            elif self._app_type == DemoMode.LSL_STRETCH:
                allowed_modes = [
                    BoardMode.WRIST_LSL_CONTINUOUS,
                    BoardMode.DPAD_WINNER_A_CONTINUOUS,
                ]
            elif self._app_type == DemoMode.DECODE_KB:
                allowed_modes = [
                    BoardMode.DIGIT_4BUTTON_DECODE_DISCRETE
                ]
            elif self._app_type == DemoMode.DECODE_MOUSE:
                allowed_modes = [
                    BoardMode.WRIST_2D_DECODE_CONTINUOUS
                ]
            else:
                # Fallback: show all modes
                allowed_modes = None

            self._viewer = BoardManagerWindow(self, allowed_modes=allowed_modes)
            self._viewer.enable_signal.connect(self.set_enabled)
            self._viewer.home_robot.connect(self._home_robot)
            self._viewer.update_mode_signal.connect(self.set_update_mode)

            # Keep the asserted checkbox synced
            self.state_change.connect(self._viewer.update_state)

        self._viewer.show()
        self._viewer.raise_()
        self._viewer.activateWindow()


    @pyqtSlot()
    def _home_robot(self):
        """Delegate homing sequence to RobotLSLHandler."""
        if self._robot is not None:
            self._robot.home()

    # ------------------------------------------------------------------
    # Simple setters
    # ------------------------------------------------------------------
    @pyqtSlot(bool)
    def set_enabled(self, en: bool):
        self._enabled = en

    @pyqtSlot(int)
    def set_assertion_debounce(self, new_debounce_duration: int):
        self._assertion_debounce_ms = new_debounce_duration

    @pyqtSlot(int)
    def set_deassertion_debounce(self, new_debounce_duration: int):
        self._deassertion_debounce_ms = new_debounce_duration

    @pyqtSlot(float)
    def set_assertion_threshold(self, new_threshold: float):
        self._assertion_threshold = new_threshold

    @pyqtSlot(float)
    def set_deassertion_threshold(self, new_threshold: float):
        self._deassertion_threshold = new_threshold

    @pyqtSlot(int)
    def set_power_threshold(self, new_threshold: int):
        self._power_threshold = new_threshold

    @pyqtSlot(float)
    def set_orientation_threshold(self, new_threshold: float):
        self._orientation_threshold = new_threshold

    # ------------------------------------------------------------------
    # Mode handling / wiring
    # ------------------------------------------------------------------
    def _wire_processor_for_mode(self, processor_mode: int) -> None:
        """
        Internal helper to connect the processor's delta_omega signal
        according to the active mode.
        """
        if processor_mode == 3:
            self.processor.delta_omega.connect(self.update_dpad_a_presses_latched)  # pyright: ignore[reportOptionalMemberAccess]
            self.processor.set_mode(3)
        elif processor_mode == 4:
            self.processor.delta_omega.connect(self.update_dpad_a_presses_continuous)
            self.processor.set_mode(3)
        elif processor_mode == 5:
            self.processor.delta_omega.connect(self.update_directed_a_only)
            self.processor.set_mode(3)
        elif processor_mode == 6:
            self.processor.delta_omega.connect(self.update_latch_toggle_a_only)
            self.processor.set_mode(3)
        elif processor_mode == 7:
            self.processor.delta_omega.connect(self.update_dpad_winner_only_a_presses_continuous)
            self.processor.set_mode(3)
        elif processor_mode == 8:
            self.processor.delta_omega.connect(self.update_lsl_robot_message)
            self.processor.set_mode(4)
        else:
            raise Exception(f"[BoardManager]::Invalid value for BoardManager._mode: {processor_mode}")

    def _unwire_processor_for_mode(self, mode: int) -> None:
        """
        Internal helper to disconnect the processor's delta_omega signal.
        """
        try:
            if mode == 3:
                self.processor.delta_omega.disconnect(self.update_dpad_a_presses_latched)
            elif mode == 4:
                self.processor.delta_omega.disconnect(self.update_dpad_a_presses_continuous)
            elif mode == 5:
                self.processor.delta_omega.disconnect(self.update_directed_a_only)
            elif mode == 6:
                self.processor.delta_omega.disconnect(self.update_latch_toggle_a_only)
            elif mode == 7:
                self.processor.delta_omega.disconnect(self.update_dpad_winner_only_a_presses_continuous)
            elif mode == 8:
                self.processor.delta_omega.disconnect(self.update_lsl_robot_message)
        except TypeError:
            # Already disconnected; ignore.
            pass

    @pyqtSlot(int)
    def set_update_mode(self, new_mode: int):
        """
        Update the mapping from processor outputs to outputs
        (gamepad vs robot), and initialize/cleanup controllers as needed.
        """
        # External UI supplies 0-based, internal modes are 3–10
        new_mode += 3
        if new_mode == self._mode:
            return  # No change

        if (new_mode < 3) or (new_mode > 10):
            raise Exception(
                f"[BoardManager]::Invalid `new_mode`: must be integer in the "
                f"range [0, 10] (sent: {new_mode})"
            )

        # Manage controller lifecycle based on previous vs new mode
        was_gamepad_mode = self._mode in (
            BoardMode.DPAD_A_LATCH,
            BoardMode.DPAD_A_CONTINUOUS,
            BoardMode.DIRECTED_A_LATCH,
            BoardMode.NONDIRECTED_A_LATCH_TOGGLE,
            BoardMode.DPAD_WINNER_A_CONTINUOUS,
        )
        will_be_gamepad_mode = new_mode in (
            BoardMode.DPAD_A_LATCH,
            BoardMode.DPAD_A_CONTINUOUS,
            BoardMode.DIRECTED_A_LATCH,
            BoardMode.NONDIRECTED_A_LATCH_TOGGLE,
            BoardMode.DPAD_WINNER_A_CONTINUOUS,
        )

        # Disconnect old wiring
        self._unwire_processor_for_mode(self._mode)

        # Handle gamepad lifecycle
        if was_gamepad_mode and not will_be_gamepad_mode and self.gamepad is not None:
            self.gamepad.cleanup()
        elif (not was_gamepad_mode) and will_be_gamepad_mode:
            # Only spin up gamepad if this app type is supposed to use it
            if self._app_type == DemoMode.VIGEM_LATCH:
                if self.gamepad is None:
                    self.gamepad = GamepadController()
                self.gamepad.initialize()

        print(f"[BoardManager]::Updating mode from {self._mode} to {BoardMode(new_mode)}")
        self._mode = BoardMode(new_mode)

        # Wire up for new mode
        self._wire_processor_for_mode(self._mode.value)

    # ------------------------------------------------------------------
    # Debounce
    # ------------------------------------------------------------------
    @pyqtSlot()
    def debounce_timeout_handler(self):
        """Handle the debounce timeout."""
        self._in_debounce = False
        self._debounce_timer.stop()

    # ------------------------------------------------------------------
    # Start/Stop
    # ------------------------------------------------------------------
    def start(self):
        """Start the QTimer only after initialization."""
        if self._running:
            print("Stream is already running!")
            return

        if self._save_logs:
            fname = f"{self.filename}_{self.suffix}.tsv"
            print(f"Logging streams to file: {fname}")
            self.board_shim.start_stream(self.buffer_size, f"file://{fname}:w")
            self.logger = BinaryLogger(f"{self.filename}_{self.suffix}.bin")
            if self.suffix is None:
                self.suffix = 0
            self.suffix = self.suffix + 1
        else:
            self.board_shim.start_stream(self.buffer_size)

        self.processor.set_filename(self.filename, self.suffix)
        self.stream_started.emit()
        self._running = True
        self.processor.start_device_sampling()  # timer to grab new samples from device
        self.processor.start_xy_interpolation()

    def stop(self):
        """Stops the streams/timers."""
        if not self._running:
            return

        self.processor.stop_xy_interpolation()
        self.processor.stop_device_sampling()
        self._running = False
        self.board_shim.stop_stream()

        if self.logger is not None:
            f = self.logger.filename
            self.logger.close()
            del self.logger
            self.logger = None
            r = BinaryReader(f)
            r.convert()
            r.close()
            del r

        if self.suffix is None:
            self.stream_stopped.emit(-1)
        else:
            self.stream_stopped.emit(self.suffix)

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    @pyqtSlot(str, int, bool)
    def on_file_handling_update(self, new_fname, new_suffix, saving_to_file):
        """Update the filename and suffix for the logger."""
        if self._running:
            self.stop()
        self._save_logs = saving_to_file
        self.filename = new_fname
        self.suffix = new_suffix
        self.filename_updated.emit(new_fname, new_suffix)

    # ------------------------------------------------------------------
    # Gamepad-based update modes
    # ------------------------------------------------------------------
    def _ensure_gamepad_ready(self) -> bool:
        if self.gamepad is None or not self.gamepad.is_ready:
            # In non-gamepad modes, these slots may still be connected
            # or invoked; just no-op safely.
            return False
        return True

    @pyqtSlot(float, float, float, float)
    def update_dpad_a_presses_latched(self, dx: float, dy: float, power: float, _):
        """
        Update the gamepad using DPad presses / fist clench for A-press,
        emitting commands only on button-combo change.
        """
        if not self._ensure_gamepad_ready():
            return

        btn = 0x0000  # Default byte code is all button bytes off.
        if dx > self._deassertion_threshold:
            btn += self._right_button_bytecode
        elif dx < -self._deassertion_threshold:
            btn += self._left_button_bytecode
        if dy > self._deassertion_threshold:
            btn += self._up_button_bytecode
        elif dy < -self._deassertion_threshold:
            btn += self._down_button_bytecode

        if power > self._power_threshold:
            btn += self._a_button_bytecode
            has_button = True
        else:
            has_button = False

        if btn != self.gamepad.buttons:
            if self._enabled:
                self.gamepad.send_input(btn)
            else:
                print(btn)
            self.state_change.emit(has_button)

    @pyqtSlot(float, float, float, float)
    def update_dpad_a_presses_continuous(self, dx: float, dy: float, power: float, _):
        """
        Update the gamepad using DPad presses / fist clench for A-press,
        emitting continuous commands.
        """
        if not self._ensure_gamepad_ready():
            return

        btn = 0x0000  # Default byte code is all button bytes off.
        if dx > self._deassertion_threshold:
            btn += self._right_button_bytecode
        elif dx < -self._deassertion_threshold:
            btn += self._left_button_bytecode
        if dy > self._deassertion_threshold:
            btn += self._up_button_bytecode
        elif dy < -self._deassertion_threshold:
            btn += self._down_button_bytecode

        if power > self._power_threshold:
            btn += self._a_button_bytecode
            self.state_change.emit(True)
        else:
            self.state_change.emit(False)

        if self._enabled:
            self.gamepad.send_input(btn)
        else:
            print(btn)

    @pyqtSlot(float, float, float, float)
    def update_dpad_winner_only_a_presses_continuous(self, dx: float, dy: float, power: float, _):
        """
        Update the gamepad using DPad presses / fist clench for A-press,
        emitting continuous commands with a 'winner-take-all' axis.
        """
        if not self._ensure_gamepad_ready():
            return

        btn = 0x0000  # Default byte code is all button bytes off.
        if (dx * dx) > (dy * dy):
            if dx > self._deassertion_threshold:
                btn += self._right_button_bytecode
            elif dx < -self._deassertion_threshold:
                btn += self._left_button_bytecode
        else:
            if dy > self._deassertion_threshold:
                btn += self._up_button_bytecode
            elif dy < -self._deassertion_threshold:
                btn += self._down_button_bytecode

        if power > self._power_threshold:
            btn += self._a_button_bytecode
            self.state_change.emit(True)
        else:
            self.state_change.emit(False)

        if self._enabled:
            self.gamepad.send_input(btn)
        else:
            print(btn)

    @pyqtSlot(float, float, float, float)
    def update_directed_a_only(self, dx: float, dy: float, _power: float, _orientation: float = 0.0):
        """
        Update the gamepad based on the angular velocity with directed return.
        """
        if not self._ensure_gamepad_ready():
            return

        if not self._in_debounce:
            if self.asserted:
                # Wait for return movement along the stored direction
                if self.return_direction == "right":
                    if dy < -self._deassertion_threshold:
                        self.asserted = False
                        if self._enabled:
                            self.gamepad.send_input(self.gamepad.buttons & (0xFFFF - self._a_button_bytecode))
                        else:
                            print("Deasserted")
                        self._in_debounce = True
                        self._debounce_timer.start(self._deassertion_debounce_ms)
                        self.state_change.emit(False)
                        self.return_direction = "NONE"
                elif self.return_direction == "left":
                    if dy > self._deassertion_threshold:
                        self.asserted = False
                        if self._enabled:
                            self.gamepad.send_input(self.gamepad.buttons & (0xFFFF - self._a_button_bytecode))
                        else:
                            print("Deasserted")
                        self._in_debounce = True
                        self._debounce_timer.start(self._deassertion_debounce_ms)
                        self.state_change.emit(False)
                        self.return_direction = "NONE"
                elif self.return_direction == "up":
                    if dx > self._deassertion_threshold:
                        self.asserted = False
                        if self._enabled:
                            self.gamepad.send_input(self.gamepad.buttons & (0xFFFF - self._a_button_bytecode))
                        else:
                            print("Deasserted")
                        self._in_debounce = True
                        self._debounce_timer.start(self._deassertion_debounce_ms)
                        self.state_change.emit(False)
                        self.return_direction = "NONE"
                elif self.return_direction == "down":
                    if dx < -self._deassertion_threshold:
                        self.asserted = False
                        if self._enabled:
                            self.gamepad.send_input(self.gamepad.buttons & (0xFFFF - self._a_button_bytecode))
                        else:
                            print("Deasserted")
                        self._in_debounce = True
                        self._debounce_timer.start(self._deassertion_debounce_ms)
                        self.state_change.emit(False)
                        self.return_direction = "NONE"
            else:
                # Detect assertion
                if (dx ** 2 > (1.5 * dy ** 2)) and (dx ** 2 > self._assertion_threshold):
                    self.asserted = True
                    if self._enabled:
                        self.gamepad.send_input(self.gamepad.buttons | self._a_button_bytecode)
                    else:
                        print("Asserted")
                    if dx < 0:
                        self.return_direction = "up"
                        print("STATE = DOWN")
                    else:
                        self.return_direction = "down"
                        print("STATE = UP")
                    self._in_debounce = True
                    self._debounce_timer.start(self._deassertion_debounce_ms)
                    self.state_change.emit(True)
                elif dy ** 2 > self._assertion_threshold:
                    self.asserted = True
                    if self._enabled:
                        self.gamepad.send_input(self.gamepad.buttons | self._a_button_bytecode)
                    else:
                        print("Asserted")
                    if dy < 0:
                        self.return_direction = "left"
                        print("STATE = RIGHT")
                    else:
                        self.return_direction = "right"
                        print("STATE = LEFT")
                    self._in_debounce = True
                    self._debounce_timer.start(self._assertion_debounce_ms)
                    self.state_change.emit(True)

    @pyqtSlot(float, float, float, float)
    def update_latch_toggle_a_only(self, dx: float, dy: float, _power: float, _orientation: float = 0.0):
        """
        Update the gamepad based on the angular velocity with latch toggle.
        """
        print(f"DX: {dx}, DY: {dy} | Power: {_power}")
        if not self._ensure_gamepad_ready():
            print("No gamepad ready.") 
            return

        if not self._in_debounce:
            if self.asserted:
                if (dx ** 2 + dy ** 2) > self._assertion_threshold:
                    self.asserted = False
                    if self._enabled:
                        self.gamepad.send_input(0x0000)
                    else:
                        print("Deasserted")
                    self._in_debounce = True
                    self._debounce_timer.start(self._deassertion_debounce_ms)
                    self.state_change.emit(False)
            else:
                if (dx ** 2 + dy ** 2) > self._assertion_threshold:
                    self.asserted = True
                    if self._enabled:
                        self.gamepad.send_input(self._a_button_bytecode)
                    else:
                        print("Asserted")
                    self._in_debounce = True
                    self._debounce_timer.start(self._assertion_debounce_ms)
                    self.state_change.emit(True)

    # ------------------------------------------------------------------
    # Robot / LSL mode
    # ------------------------------------------------------------------
    @pyqtSlot(float, float, float, float)
    def update_lsl_robot_message(self, dx: float, dy: float, power: float, orientation: float):
        """
        Delegate robot updates to RobotLSLHandler.
        """
        if self._robot is None:
            return

        self._robot.update(
            dx=dx,
            dy=dy,
            power=power,
            orientation=orientation,
            assertion_threshold=self._assertion_threshold,
            power_threshold=self._power_threshold,
            orientation_threshold=self._orientation_threshold,
            enabled=self._enabled,
            state_change_cb=self.state_change.emit,
        )
