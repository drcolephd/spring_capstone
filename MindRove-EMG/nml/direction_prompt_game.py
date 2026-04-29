from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import random
import numpy as np

class DirectionPromptGame(QtWidgets.QWidget):
    marker = pyqtSignal(float, name='marker') 
    closed = pyqtSignal()

    start_button = None
    stop_button = None
    canvas = None
    _arrow_present = False

    def __init__(self, add_buttons: bool = True):
        super(DirectionPromptGame, self).__init__()
        self.setWindowTitle("Direction Prompt Game")
        self.setStyleSheet("background-color: black;")
        self.setGeometry(950, 100, 800, 900)

        # Create a grid layout
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(5)
        
        # Create Start and Stop buttons
        if add_buttons:
            self.start_button = QtWidgets.QPushButton("Start")
            self.start_button.setStyleSheet("""
                QPushButton:enabled {
                    background-color: white;
                    color: blue;
                }
                QPushButton:disabled {
                    background-color: lightgray;
                    color: gray;
                }
            """)
            self.stop_button = QtWidgets.QPushButton("Stop")
            self.stop_button.setStyleSheet("""
                QPushButton:enabled {
                    background-color: white;
                    color: red;
                }
                QPushButton:disabled {
                    background-color: lightgray;
                    color: gray;
                }
            """)
            self.stop_button.setEnabled(False)
            layout.addWidget(self.start_button, 0, 1, 1, 1)
            layout.addWidget(self.stop_button, 0, 3, 1, 1)
            # Connect button signals
            self.start_button.clicked.connect(self.start)
            self.stop_button.clicked.connect(self.stop)
        
        # Game canvas area
        self.canvas = QtWidgets.QWidget(self)
        self.canvas.setStyleSheet("background-color: black;")
        layout.addWidget(self.canvas, 1, 0, 4, 5)
        
        # Timer for showing arrows
        self.arrow_timer = QtCore.QTimer()
        self.arrow_timer.setSingleShot(True)
        self.arrow_timer.timeout.connect(self.update_and_restart_timer)
        self.direction = -1
        self.current_arrow = None

    @staticmethod
    def _generate_exponential_duration(min_time=1000, max_time=3500, scale=500):
        """
        Generate a random time duration based on an exponential distribution,
        clamped between min_time and max_time.

        Parameters:
        - min_time: Minimum time in milliseconds.
        - max_time: Maximum time in milliseconds.
        - scale: Scale factor for the exponential distribution (affects the mean).

        Returns:
        - duration: Clamped duration in milliseconds.
        """
        # Generate a random duration from an exponential distribution
        duration = np.random.exponential(scale)
        
        # Clamp the duration between min_time and max_time
        duration = int(np.clip(duration, min_time, max_time))
        
        return duration

    def update_and_restart_timer(self):
        """Function to update the timer period dynamically."""
        if self._arrow_present:
            self.clear_arrows()
            self._arrow_present = False
            self.marker.emit(1) # Indicate it is "REST" epoch
        else:
            self.show_arrow()
        new_interval = DirectionPromptGame._generate_exponential_duration()  # Set your new interval in milliseconds
        self.arrow_timer.start(new_interval)

    def start(self):
        """Start the game and show arrows at intervals."""
        if self.start_button is not None:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        self.marker.emit(11)        
        new_interval = DirectionPromptGame._generate_exponential_duration()  # Set your new interval in milliseconds
        self.arrow_timer.start(new_interval)

    def stop(self):
        """Stop the game and hide arrows."""
        if self.stop_button is not None:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        self.arrow_timer.stop()
        self.clear_arrows()
        self.marker.emit(12)

    def show_arrow(self):
        """Display an arrow at a random direction."""
        self.clear_arrows()
        
        # Directions: 0 = Top, 1 = Right, 2 = Bottom, 3 = Left
        self.direction = random.randint(0, 3)
        
        arrow_label = QtWidgets.QLabel(self.canvas)
        arrow_label.setStyleSheet("color: white; width: 200px; height: 200px")
        
        if self.direction == 0:
            arrow_label.setText("↑")
            arrow_label.move(self.canvas.width() // 2 - 10, 15)
        elif self.direction == 1:
            arrow_label.setText("→")
            arrow_label.move(self.canvas.width() - 50, self.canvas.height() // 2 - 10)
        elif self.direction == 2:
            arrow_label.setText("↓")
            arrow_label.move(self.canvas.width() // 2 - 10, self.canvas.height() - 50)
        elif self.direction == 3:
            arrow_label.setText("←")
            arrow_label.move(15, self.canvas.height() // 2 - 10)
        arrow_label.setFont(QtGui.QFont("Arial", 24))
        arrow_label.show()
        self.current_arrow = arrow_label
        self.marker.emit(self.direction+2)
        self._arrow_present = True

    @pyqtSlot(int)
    def on_key_press(self, val):
        if self.current_arrow is None:
            return
        if (val-7) == self.direction:
            self.current_arrow.setText("✓")
        else:
            self.current_arrow.setText("X")
        self.current_arrow.show()
        self.arrow_timer.stop()
        new_interval = DirectionPromptGame._generate_exponential_duration()
        self.arrow_timer.start(new_interval)

    def closeEvent(self, event):
        event.accept()
        self.closed.emit()

    def clear_arrows(self):
        """Remove any displayed arrow from the canvas."""
        if self.current_arrow is not None:
            self.current_arrow.deleteLater()
            del self.current_arrow
            self.current_arrow = None
