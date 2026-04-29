from PyQt5 import QtWidgets, QtCore, QtGui

class GoalCircle(QtCore.QObject):
    consumed = QtCore.pyqtSignal(int)  # Define the consumed signal

    def __init__(self, grid, row, column, value):
        super().__init__()  # Initialize QObject
        self.grid_ = grid
        self.row = row
        self.column = column
        self.value = value
        self.available = True
        # Create a QGraphicsEllipseItem instance as an attribute for visual representation
        self.ellipse_item = QtWidgets.QGraphicsEllipseItem(0, 0, 5, 5)
        self.ellipse_item.setBrush(QtGui.QBrush(QtGui.QColor("green")))  # Set color to yellow
        initial_pos = self.grid_.cell(self.row, self.column, 5, 5)
        self.ellipse_item.setPos(initial_pos)
        
    def add_to_scene(self, scene):
        scene.addItem(self.ellipse_item)
    
    def consume(self):
        """Consume the goal circle and emit the consumed signal."""
        self.consumed.emit(self.value)  # Emit the signal
        self.ellipse_item.hide()  # Hide the goal circle
        self.available = False

