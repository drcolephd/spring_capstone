from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from nml.pacman_level import PacmanLevel

class PacmanGame(QtWidgets.QWidget):
    state = pyqtSignal(int,  name='pacman_game')
    closed = pyqtSignal() 

    def __init__(self, difficulty=0):
        super().__init__()
        self.setWindowTitle("Pacman Game")
        self.setStyleSheet("background-color: black;")
        self.setGeometry(300, 100, 800, 800)  # Window size and position

        # Create the PacmanLevel game area
        self.level = PacmanLevel(difficulty)
        self.level.complete.connect(self.onLevelCompleted)
        self.level.collision.connect(self.onLevelCollision)
        
        # Create Start and Stop buttons
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
        
        # Arrange buttons and level in a vertical layout
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(5)
        layout.addWidget(self.start_button, 0, 0, 1, 1)
        layout.addWidget(self.stop_button, 0, 1, 1, 1)
        layout.addWidget(self.level, 1, 0, 2, 2)
        self.level.fitInView(self.level.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
        self.setLayout(layout)

        # Connect button signals
        self.start_button.clicked.connect(self.handleGameStartButtonClick)
        self.stop_button.clicked.connect(self.handleGameStopButtonClick)

    def handleGameStartButtonClick(self):
        """Initialize or restart the game."""
        print("Game started")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.state.emit(15)
        self.level.start_game()

    def handleGameStopButtonClick(self):
        """Stop the game and reset elements as needed."""
        print("Game stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.level.stop_game()
        self.state.emit(21)

    def resizeEvent(self, event):
        """Handle resize to ensure the level fits the window."""
        super().resizeEvent(event)
        self.level.fitInView(self.level.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)

    def closeEvent(self, event):
        event.accept()
        self.closed.emit()

    @pyqtSlot(int)
    def onLevelCollision(self, collisionType):
        """Handle collision between Pacman and elements in game."""
        if collisionType==0: 
            print("Pacman collided with a Ghost!")
            self.state.emit(18)
        elif collisionType==1:
            print("Pacman consumed a Pellet!")
            self.state.emit(15)

    @pyqtSlot(int)
    def onLevelCompleted(self):
        print("PacMan Level complete!")
        self.state.emit(19)
        self.level.show_message("Level Complete!")
        QtCore.QTimer.singleShot(1500, lambda: self.close())

