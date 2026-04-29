import os
import datetime
from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLineEdit, QCheckBox, QLabel, QSpinBox, QGridLayout, QComboBox
)
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QTimer, QEvent, Qt
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams
from nml.utilities import get_mindrove_ip
from nml.realtime_plotter import RealTimePlotter  
from nml.direction_prompt_game import DirectionPromptGame
from nml.teensy import Teensy as Manip
from nml.serial_plot_window import SerialPlotWindow
from nml.pacman_game import PacmanGame
from nml.local_paths import paths
from nml.connections.tcp import TCP
from nml.connections.ws import WebSocketServer
from nml.gesture_calibration_window import GestureCalibrationWindow
import logging, os, subprocess
class Application(QMainWindow):
    control = pyqtSignal(int,  name='control')
    marker = pyqtSignal(float, name='marker')
    name = pyqtSignal(str, int, bool, name='filename')
    
    manip:  Manip | None = None
    board_shim: BoardShim | None = None
    board_id: BoardIds | None = None
    mouse: TCP | None = None
    ws: WebSocketServer | None = None

    filename_input: QLineEdit | None = None
    save_checkbox: QCheckBox | None = None
    suffix_spinbox: QSpinBox | None = None
    open_plot_button: QPushButton | None = None
    direction_game_button: QPushButton | None = None
    pacman_game_button: QPushButton | None = None
    pacman_difficulty_spinbox: QSpinBox | None = None
    control_dropdown: QComboBox | None = None
    connect_button: QPushButton | None = None
    monitor_button: QPushButton | None = None
    gesture_cal_button: QPushButton | None = None

    real_time_plotter: RealTimePlotter | None = None
    gesture_cal_window: GestureCalibrationWindow | None = None
    pacman_game: PacmanGame | None = None
    direction_prompt_game: DirectionPromptGame | None = None
    serial_monitor: SerialPlotWindow | None = None

    control_method: str = "Keyboard"  # Default control method
    pacman_difficulty: int = 0

    _icon_image_file: str = "cmu-scotty-scarf.png"
    _background_image_file: str = "cmu-tartan-wave-gray-crop-01.png"
    _mouse_executable_path: str = os.path.join(paths['root'], "mouse.exe")
    _default_save_status: bool = False
    _debounce_timer: QTimer = QTimer()
    _debounce_interval: int = 100

    def __init__(self, app, args):
        super(Application, self).__init__()
        self.app = app
        self.args = args

        # Initialize logging and BoardShim
        BoardShim.enable_dev_board_logger()
        logging.basicConfig(level=logging.DEBUG)
        if args.synth==0:
            self.board_id = BoardIds.MINDROVE_WIFI_BOARD
        else:
            self.board_id = BoardIds.SYNTHETIC_BOARD
        params = MindRoveInputParams()
        self.board_shim = BoardShim(self.board_id, params)
        s = BoardShim.get_board_descr(self.board_id)
        print(s)
        # self.board_shim.release_all_sessions()
        self.board_shim.prepare_session()
        # params.ip_address = get_mindrove_ip()
        # print(params.to_json())

        self.ws = WebSocketServer(self.board_shim)
        self.ws.start()

        # Set up main window layout
        self.setWindowIcon(QtGui.QIcon(os.path.join(paths['assets'], self._icon_image_file)))
        self.setWindowTitle("NML-MindRove Main GUI")
        self.setGeometry(100, 100, 800, 300)
        self._initialize_cmu_style()

        # Define layout
        central_widget = QWidget()
        layout = QGridLayout()

        # Filename Label and Input
        today_str = datetime.datetime.now().strftime('%Y_%m_%d')
        default_filename = f"data/{self.args.file}_{today_str}"
        
        layout.addWidget(QLabel("Filename:"), 0, 0)
        self.filename_input = QLineEdit(default_filename)
        layout.addWidget(self.filename_input, 0, 1, 1, 3)
        self.filename_input.textChanged.connect(self.handle_file_status_update)

        # Save Checkbox
        self.save_checkbox = QCheckBox("Save to File")
        self.save_checkbox.setChecked(self._default_save_status)
        layout.addWidget(self.save_checkbox, 1, 0)
        self.save_checkbox.clicked.connect(self.handle_file_status_update)

        # Suffix Increment
        layout.addWidget(QLabel("File Suffix:"), 1, 1)
        self.suffix_spinbox = QSpinBox()
        self.suffix_spinbox.setMinimum(0)
        self.suffix_spinbox.setValue(args.suffix)
        layout.addWidget(self.suffix_spinbox, 1, 2)
        self.suffix_spinbox.valueChanged.connect(self.handle_file_status_update)

        # Mouse Emulator
        self.mouse_button = QPushButton("Emulate Mouse", self)
        layout.addWidget(self.mouse_button , 1, 3)
        self.mouse_button.clicked.connect(self._emulate_mouse)

        # Button to open RealTimePlotter
        self.open_plot_button = QPushButton("Open Streams", self)
        self.open_plot_button.clicked.connect(self.handle_open_real_time_plotter)
        layout.addWidget(self.open_plot_button, 3, 0, 1, 2)

        # DirectionPromptGame button (initially disabled)
        self.direction_game_button = QPushButton("Open Arrows Game")
        # self.direction_game_button.setEnabled(False)
        layout.addWidget(self.direction_game_button, 3, 2, 1, 2)
        self.direction_game_button.clicked.connect(self.handle_open_direction_prompt_game)

        # PacmanGame button
        self.pacman_game_button = QPushButton("Open PacMan Game")
        layout.addWidget(self.pacman_game_button, 4, 2, 1, 2)
        self.pacman_game_button.clicked.connect(self.handle_open_pacman_game)

        # Pacman Difficulty
        layout.addWidget(QLabel("Difficulty:"), 4, 0, 1, 1)
        self.pacman_difficulty_spinbox = QSpinBox()
        self.pacman_difficulty_spinbox.setMinimum(0)
        self.pacman_difficulty_spinbox.setValue(0)
        layout.addWidget(self.pacman_difficulty_spinbox, 4, 1, 1, 1)

        # Gesture Calibration button (requires stream to be open)
        self.gesture_cal_button = QPushButton("Gesture Calibration")
        self.gesture_cal_button.setEnabled(False)
        self.gesture_cal_button.clicked.connect(self.handle_open_gesture_calibration)
        layout.addWidget(self.gesture_cal_button, 5, 0, 1, 4)

        # Add control method dropdown
        layout.addWidget(QLabel("Controller:"), 2, 0, 1, 1)
        self.control_dropdown = QComboBox()
        self.control_dropdown.addItem("Keyboard")
        self.control_dropdown.addItem("MindRove Board")
        self.control_dropdown.currentIndexChanged.connect(self.handle_control_method_change)
        layout.addWidget(self.control_dropdown, 2, 1, 1, 1)

        self.connect_button = QPushButton("Connect Manip", self)
        layout.addWidget(self.connect_button, 2, 2, 1, 1)
        self.connect_button.clicked.connect(self.handle_connect_to_serial)

        self.monitor_button = QPushButton("Show Manip", self)
        layout.addWidget(self.monitor_button , 2, 3, 1, 1)
        self.monitor_button.clicked.connect(self._raise_serial_monitor)

        # Set up central widget with the grid layout
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Setup event filter
        self.app.installEventFilter(self)  # Install on the application instance
        self.marker.connect(self.on_marker_signal)
        self._debounce_timer.setSingleShot(True)

    def _run_mouse(self):
        """
        Runs the mouse.exe executable.
        """
        try:
            # Construct the command
            command = [self._mouse_executable_path]

            # Run the executable (non-blocking)
            _ = subprocess.Popen(command)
            self.mouse = TCP(ip="127.0.0.1", mouse_port = 6054)
            if self.real_time_plotter is not None:
                self.real_time_plotter.processor.mouse.connect(self.mouse.handle_mouse)
            
        except subprocess.CalledProcessError as e:
            print(f"Executable failed with error code {e.returncode}")
            print(f"Output: {e.stdout}")
            print(f"Errors: {e.stderr}")
        except FileNotFoundError:
            print(f"Executable not found: {self._mouse_executable_path}")

    def _emulate_mouse(self):
        self.mouse_button.clicked.disconnect(self._emulate_mouse)
        self.mouse_button.setText("Stop Mouse")
        self.mouse_button.clicked.connect(self._stop_mouse)
        self._run_mouse()

    def _stop_mouse(self):
        self.mouse_button.clicked.disconnect(self._stop_mouse)
        self.mouse_button.setText("Emulate Mouse")
        self.mouse_button.clicked.connect(self._emulate_mouse)
        self.mouse.close()
        if self.real_time_plotter is not None:
            self.real_time_plotter.processor.mouse.disconnect(self.mouse.handle_mouse)
        del self.mouse
        self.mouse = None

    def closeEvent(self, event):
        """Ensure resources are properly cleaned up when the main application closes."""
        if self.real_time_plotter is not None:
            self.real_time_plotter.close()  # Trigger RealTimePlotter cleanup
        event.accept()

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress and not self._debounce_timer.isActive():
            # Start debounce timer to prevent multiple events
            self._debounce_timer.start(self._debounce_interval)
            if event.key() == Qt.Key_Space:
                print("Spacebar pressed: Marker = 1")
                self.marker.emit(1)
            elif event.key() == Qt.Key_Return:
                print("Enter pressed: Marker = 2")
                self.marker.emit(2)
            elif event.key() == Qt.Key_Escape:
                print("Escape pressed: exiting application")
                self.close()
            elif event.key() == Qt.Key_Up:
                print("Up")
                self.marker.emit(7)
                if self.control_method == "Keyboard":
                    self.control.emit(7)
            elif event.key() == Qt.Key_Down:
                print("Down")
                if self.control_method == "Keyboard":
                    self.control.emit(9)
                self.marker.emit(9)
            elif event.key() == Qt.Key_Left:
                print("Left")
                if self.control_method == "Keyboard":
                    self.control.emit(10)
                self.marker.emit(10)
            elif event.key() == Qt.Key_Right:
                print("Right")
                if self.control_method == "Keyboard":
                    self.control.emit(8)
                self.marker.emit(8)
        return super().eventFilter(source, event)
    
    def handle_connect_to_serial(self):
        if self.manip is None:
            self.manip = Manip(n_channels=2)
        self.manip.connected.connect(self._on_serial_connection)
        self.manip.disconnected.connect(self._on_disconnect_from_serial)
        self.manip.connect()

    def _on_serial_connection(self):
        self.manip.start()
        self._raise_serial_monitor()
        # Connect Teensy's newData signal to the plot window's update method
        self.manip.newData.connect(self.serial_monitor.update_plot)
        if self.real_time_plotter is not None:
            self.manip.newData.connect(self.real_time_plotter.on_new_manip_data)
        self.connect_button.clicked.disconnect(self.handle_connect_to_serial)
        self.connect_button.clicked.connect(self.handle_disconnect_from_serial)
        self.connect_button.setText("Disconnect Manip")

    def _raise_serial_monitor(self):
        if self.serial_monitor is None:
            self.serial_monitor = SerialPlotWindow(n_channels=2, buffer_size=300, refresh_rate=30.0)
            self.serial_monitor.closed.connect(self._on_serial_monitor_close)
        self.serial_monitor.show()
        self.serial_monitor.raise_()
        self.monitor_button.setEnabled(False)

    def _on_serial_monitor_close(self):
        self.monitor_button.setEnabled(True)

    def handle_disconnect_from_serial(self):
        self.manip.disconnect()

    def _on_disconnect_from_serial(self):
        self.connect_button.clicked.disconnect(self.handle_disconnect_from_serial)
        self.connect_button.clicked.connect(self.handle_connect_to_serial)
        self.connect_button.setText("Connect Manip")
        if self.serial_monitor is not None:
            self.manip.newData.disconnect(self.serial_monitor.update_plot)
        if self.real_time_plotter is not None:
            self.manip.newData.disconnect(self.real_time_plotter.on_new_manip_data)
        self.manip.connected.disconnect(self._on_serial_connection)
        self.manip.disconnected.disconnect(self._on_disconnect_from_serial)
    
    def handle_file_status_update(self):
        """Emit signal indicating new file handling (filename, suffix, boolean of whether ot save or not)."""
        self.name.emit(self.filename_input.text(), self.suffix_spinbox.value(), self.save_checkbox.isChecked()) # type: ignore

    def handle_open_real_time_plotter(self):
        """Open or show the RealTimePlotter window as a sub-window."""
        if self.real_time_plotter is None:
            # Check if saving is enabled
            filename = self.filename_input.text()
            suffix = self.suffix_spinbox.value()
            # Initialize RealTimePlotter if it doesn't exist or is hidden
            self.real_time_plotter = RealTimePlotter(self.app, self.board_shim, is_main_window=False, save_logs=self.save_checkbox.isChecked(), filename=filename, suffix=suffix)
            if self.manip is not None:
                self.manip.newData.connect(self.real_time_plotter.on_new_manip_data)
            self.real_time_plotter.closed.connect(self.on_plotter_closing)
            self.real_time_plotter.stream_started.connect(self.on_stream_started)
            self.real_time_plotter.stream_stopped.connect(self.on_stream_stopped)
            self.real_time_plotter.block_changed.connect(self.on_stream_stopped)
            self.real_time_plotter.marker.connect(self.on_marker_signal)
            self.name.connect(self.real_time_plotter.on_file_handling_update) # type: ignore
            # self.direction_game_button.setEnabled(True)
        
        if self.mouse is not None:
            self.real_time_plotter.processor.mouse.connect(self.mouse.handle_mouse)
        self.open_plot_button.setEnabled(False)
        self.real_time_plotter.show()  # Show the plot window

    @pyqtSlot()
    def on_plotter_closing(self):
        """Disable the game button when the RealTimePlotter is closed."""
        # self.direction_game_button.setEnabled(False)
        self.real_time_plotter.stream_started.disconnect(self.on_stream_started)
        self.real_time_plotter.stream_stopped.disconnect(self.on_stream_stopped)
        self.real_time_plotter.closed.disconnect(self.on_plotter_closing)
        if self.mouse is not None:
            self.real_time_plotter.processor.mouse.disconnect(self.mouse.handle_mouse)
        if self.gesture_cal_window is not None:
            self.gesture_cal_window.close()
            self.gesture_cal_window = None
        self.gesture_cal_button.setEnabled(False)
        del self.real_time_plotter
        self.real_time_plotter = None
        self.open_plot_button.setEnabled(True)

    @pyqtSlot()
    def on_stream_started(self):
        self.gesture_cal_button.setEnabled(True)

    @pyqtSlot(int)
    def on_stream_stopped(self, val):
        self.gesture_cal_button.setEnabled(False)
        if val > -1:
            self.suffix_spinbox.setValue(val)

    def handle_open_gesture_calibration(self):
        """Open or raise the GestureCalibrationWindow."""
        if self.real_time_plotter is None:
            return
        if self.gesture_cal_window is None:
            self.gesture_cal_window = GestureCalibrationWindow(
                self.real_time_plotter.processor
            )
            self.gesture_cal_window.gesture_model_trained.connect(
                self.real_time_plotter.processor.set_gesture_model
            )
            self.gesture_cal_window.closed.connect(self._on_gesture_cal_closed)
        self.gesture_cal_window.show()
        self.gesture_cal_window.raise_()

    @pyqtSlot()
    def _on_gesture_cal_closed(self):
        self.gesture_cal_window = None

    def handle_open_direction_prompt_game(self):
        """Open the Direction Prompt Game window."""
        if self.direction_prompt_game is None:
            self.direction_prompt_game = DirectionPromptGame()
            self.direction_prompt_game.marker.connect(self.on_marker_signal)
            self.direction_prompt_game.closed.connect(self.on_direction_prompt_game_closed)
            self.control.connect(self.direction_prompt_game.on_key_press)
        self.direction_prompt_game.show()
        self.direction_game_button.setEnabled(False)

    @pyqtSlot()
    def on_direction_prompt_game_closed(self):
        self.direction_prompt_game.marker.disconnect(self.on_marker_signal)
        self.direction_prompt_game.closed.disconnect(self.on_direction_prompt_game_closed)
        del self.direction_prompt_game
        self.direction_prompt_game = None
        self.direction_game_button.setEnabled(True)

    def handle_open_pacman_game(self):
        """Open the PacMan Game window."""
        if self.pacman_game is None:
            self.pacman_game = PacmanGame(self.pacman_difficulty_spinbox.value())
            self.control.connect(self.pacman_game.level.pacman.move)
            self.pacman_game.state.connect(self.on_pacman_state_signal)
            self.pacman_game.closed.connect(self.on_pacman_game_closing)
            self.pacman_game.show()
        self.pacman_game_button.setEnabled(False)

    @pyqtSlot()
    def on_pacman_game_closing(self):
        self.pacman_game.state.disconnect(self.on_pacman_state_signal)
        self.pacman_game.closed.disconnect(self.on_pacman_game_closing)
        del(self.pacman_game)
        self.pacman_game = None
        self.pacman_game_button.setEnabled(True)

    def handle_control_method_change(self):
        """Handle changes in control method selection."""
        self.control_method = self.control_dropdown.currentText()
        print(f"Control method changed to: {self.control_method}")
        if self.pacman_game:
            self.pacman_game.set_control_method(self.control_method)

    @pyqtSlot(float)
    def on_marker_signal(self, val):
        self.board_shim.insert_marker(val)
        print(f"received: {val}")

    @pyqtSlot(int)
    def on_pacman_state_signal(self, val):
        if self.board_shim is not None:
            self.board_shim.insert_marker(float(val))
        if val == 19: # Then we completed the current level. Increment difficulty by 1.
            self.pacman_difficulty_spinbox.setValue(self.pacman_difficulty_spinbox.value()+1)

    def closeEvent(self, event):
        event.accept()
        if self.mouse is not None:
            self._stop_mouse()
        self.app.quit()
        
    def __del__(self):
        try:
            if (self.board_shim is not None) and (self.real_time_plotter is not None or not self.real_time_plotter.isHidden()):
                    if self.board_shim.is_prepared():
                        self.board_shim.release_session()  # Clean up the board session
        except Exception:
            pass
        finally:
            print("Session exited successfully.")

    def _initialize_cmu_style(self):
        """Use CMU theme colors to make the styling nicer."""
        CMU_RED = "#A6192E"
        CMU_GRAY = "#58585B"
        CMU_DARK_GRAY = "#2D2926"
        CMU_WHITE = "#FFFFFF"

        # Setting the stylesheet in Application's init method
        image_file = os.path.join(paths['assets'], self._background_image_file).replace('\\','/')
        self.setStyleSheet(f"""
            QMainWindow {{
                background-image: url({image_file});
                background-repeat: no-repeat;
                background-position: center;
                background-attachment: fixed;
                background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent overlay */
            }}

            QLabel, QLineEdit, QCheckBox, QPushButton, QSpinBox, QComboBox {{
                background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent for readability */
                color: {CMU_GRAY}; /* CMU gray */
                border: 1px solid {CMU_RED}; /* CMU red for a subtle border */
                font-size: 14px;
                padding: 4px;
                border-radius: 4px;
            }}

            /* Hover effect for QPushButton */
            QPushButton:hover {{
                background-color: {CMU_RED};
                color: {CMU_WHITE}; /* White text on hover */
                border: 1px solid {CMU_DARK_GRAY}; /* Darker border on hover */
            }}

            QPushButton:disabled {{
                background-color: {CMU_GRAY};
                color: {CMU_WHITE};
            }}

            /* Hover effect for QLineEdit */
            QLineEdit:hover {{
                border: 1px solid {CMU_GRAY}; /* CMU Gray border on hover */
                background-color: rgba(255, 255, 255, 0.9); /* Slightly more opaque background */
            }}

            /* Hover effect for QCheckBox */
            QCheckBox:hover {{
                color: {CMU_RED}; /* Change checkbox text color on hover */
            }}
            QCheckBox::indicator:hover {{
                background-color: {CMU_RED}; /* Highlight checkbox indicator on hover */
                border: 1px solid {CMU_DARK_GRAY};
            }}

            /* Hover effect for QComboBox */
            QComboBox:hover {{
                border: 1px solid {CMU_RED}; /* Highlight border on hover */
                background-color: rgba(255, 255, 255, 0.9); /* Slightly more opaque background */
            }}

            /* Hover effect for QSpinBox */
            QSpinBox:hover {{
                border: 1px solid {CMU_GRAY}; /* CMU Gray border on hover */
                background-color: rgba(255, 255, 255, 0.9); /* Slightly more opaque background */
            }}
        """)
