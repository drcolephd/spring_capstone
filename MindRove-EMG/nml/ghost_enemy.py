from PyQt5 import QtWidgets, QtCore, QtGui

class GhostEnemy(QtCore.QObject):
    exploded = QtCore.pyqtSignal()  

    def __init__(self, grid, row, column, speed_threshold=10):
        super().__init__()  # Initialize QObject
        self.grid_ = grid
        self.row = row
        self.column = column
        self.speed_threshold = speed_threshold  # Set movement speed
        self.speed_counter = 0  # Initialize counter for speed control

        # Create a QGraphicsEllipseItem instance as an attribute for visual representation
        self.ellipse_item = QtWidgets.QGraphicsEllipseItem()
        self.ellipse_item.setRect(0, 0, 10, 10)  # Fixed size for Ghost
        self.ellipse_item.setBrush(QtGui.QBrush(QtGui.QColor("red")))  # Set color to red

        # Set initial position on the scene based on grid coordinates
        initial_pos = self.grid_.cell(self.row, self.column)
        self.ellipse_item.setPos(initial_pos)

        # Attributes for explosion animation
        self.explosion_timer = QtCore.QTimer()
        self.explosion_timer.timeout.connect(self._animate_explosion)
        self.flash_counter = 0
        self.max_flashes = 6
        self.expansion_step = 1.3  # Growth rate per frame
        self.current_radius = 10  # Initial radius for ghost

    def add_to_scene(self, scene):
        scene.addItem(self.ellipse_item)

    def chase_pacman(self, pacman_position):
        """Move ghost towards Pacman using grid-based movement with speed control."""
        self.speed_counter += 1
        if self.speed_counter < self.speed_threshold:
            return

        self.speed_counter = 0

        # Convert Pacman's position to grid coordinates
        pacman_grid_x = int(pacman_position.x() / (self.grid_.cell_size + self.grid_.border_thickness))
        pacman_grid_y = int(pacman_position.y() / (self.grid_.cell_size + self.grid_.border_thickness))

        # Calculate direction to move towards Pacman
        new_row, new_column = self.row, self.column
        if pacman_grid_x > self.column:
            new_column += 1  # Move right
        elif pacman_grid_x < self.column:
            new_column -= 1  # Move left
        if pacman_grid_y > self.row:
            new_row += 1  # Move down
        elif pacman_grid_y < self.row:
            new_row -= 1  # Move up

        # Update the position if no boundary blocks movement
        new_pos, self.row, self.column = self.grid_.move(self.row, self.column, new_row, new_column)
        self.ellipse_item.setPos(new_pos)

    def explode(self):
        """Start the explosion animation sequence."""
        self.flash_counter = 0
        self.current_radius = 20
        self.explosion_timer.start(100)  # Trigger explosion every 100 ms

    def _animate_explosion(self):
        """Animate the explosion effect with flashing and expansion."""
        if self.flash_counter >= self.max_flashes:
            # Stop timer and hide the ghost after the final explosion
            self.explosion_timer.stop()
            self.ellipse_item.hide()
            self.exploded.emit()
            return

        # Flash color between red and white
        if self.flash_counter % 2 == 0:
            self.ellipse_item.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        else:
            self.ellipse_item.setBrush(QtGui.QBrush(QtGui.QColor("red")))

        # Update the explosion growth effect
        self.current_radius *= self.expansion_step
        self.ellipse_item.setRect(-self.current_radius / 2, -self.current_radius / 2,
                                  self.current_radius, self.current_radius)

        # Create a temporary black expanding circle (background explosion effect)
        black_circle = QtWidgets.QGraphicsEllipseItem(self.ellipse_item.rect())
        black_circle.setBrush(QtGui.QBrush(QtGui.QColor("black")))
        black_circle.setOpacity(0.5)
        black_circle.setZValue(self.ellipse_item.zValue() - 1)  # Place it behind the ghost

        # Add to scene and schedule removal
        scene = self.ellipse_item.scene()
        if scene:
            scene.addItem(black_circle)
            QtCore.QTimer.singleShot(500, lambda: scene.removeItem(black_circle))

        self.flash_counter += 1
