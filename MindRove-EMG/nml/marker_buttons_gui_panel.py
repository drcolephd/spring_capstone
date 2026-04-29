from PyQt5 import QtWidgets, QtCore
from pyqtgraph.Qt import QtGui
from nml.gui_window import GuiWindow
from nml.local_paths import paths
import os

class MarkerButtonsWindow(GuiWindow):
    markerClicked = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Marker Buttons")
        self.setGeometry(200, 200, 300, 200)
        self.setWindowIcon(QtGui.QIcon(os.path.join(paths['assets'], "MarkerButtonsWindowIcon.png")))
        self._initialize_marker_buttons()

    def _initialize_marker_buttons(self):
        button_stylesheet = """
            QPushButton:enabled {
                background-color: white;
                color: black;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: lightgray;
                color: gray;
            }
        """
        # Create buttons for markers 0 through 9 and add them to the grid layout
        self.marker_buttons = []
        for i in range(10):
            button = QtWidgets.QPushButton(str(i), self)
            button.setStyleSheet(button_stylesheet)
            button.clicked.connect(lambda _, i=i: self.markerClicked.emit(30.0 + i))
            self.layout.addWidget(button, i // 5, i % 5)
            self.marker_buttons.append(button)
        self.setLayout(self.layout)
