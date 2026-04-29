from PyQt5 import QtWidgets, QtCore, QtGui

class PacmanPlayer(QtCore.QObject):
    def __init__(self, grid, row, column):
        super().__init__()  # Initialize QObject
        self.grid_ = grid
        self.row = row
        self.column = column

        # Create a QGraphicsEllipseItem instance as an attribute for visual representation
        self.ellipse_item = QtWidgets.QGraphicsEllipseItem()
        self.ellipse_item.setRect(0, 0, 10, 10)  # Fixed size for Pacman
        self.ellipse_item.setBrush(QtGui.QBrush(QtGui.QColor("yellow")))  # Set color to yellow

        # Set initial position on the scene based on grid coordinates
        initial_pos = self.grid_.cell(self.row, self.column)
        self.ellipse_item.setPos(initial_pos)

    def add_to_scene(self, scene):
        scene.addItem(self.ellipse_item)

    @QtCore.pyqtSlot(int)
    def move(self, direction):
        """Move Pacman in a given direction, using the grid for cell-based movement."""
        # Determine new row and column based on direction
        if direction == 7:  # "up"
            new_row, new_col = self.row - 1, self.column
        elif direction == 9:  # "down"
            new_row, new_col = self.row + 1, self.column
        elif direction == 10:  # "left"
            new_row, new_col = self.row, self.column - 1
        elif direction == 8:  # "right"
            new_row, new_col = self.row, self.column + 1
        else:
            # Invalid direction; do nothing
            return

        # Use the grid's move method to get the final position considering walls and boundaries
        new_pos, final_row, final_col = self.grid_.move(self.row, self.column, new_row, new_col)

        # Update Pacman's position if it has changed
        if (final_row, final_col) != (self.row, self.column):
            self.ellipse_item.setPos(new_pos)
            self.row, self.column = final_row, final_col  # Update Pacman's internal position
