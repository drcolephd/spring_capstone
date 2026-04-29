from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
from nml.pacman_player import PacmanPlayer
from nml.goal_circle import GoalCircle
from nml.ghost_enemy import GhostEnemy
from nml.game_text import GameText
from nml.grid import Grid

class PacmanLevel(QtWidgets.QGraphicsView):
    complete = QtCore.pyqtSignal(int)
    collision = QtCore.pyqtSignal(int)

    def __init__(self, difficulty=0):
        super().__init__()
        self.difficulty = difficulty
        self.grid = Grid(51, 51, 10, 1)  # Initialize grid with specified cell size and border
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.score = 0
        self.score_threshold_ = difficulty*10 + 20
        self.running_ = False

        # Prevent scrollbars
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # Add Pacman, Ghost, and pellets
        self.pacman = PacmanPlayer(self.grid, row=10, column=10)
        self.ghost = GhostEnemy(self.grid, row=15, column=15)
        self.pellets = [GoalCircle(self.grid, row, col, value=10) for row, col in [(5, 5), (7, 10), (10, 15)]]

        # Add items to the scene
        self.pacman.add_to_scene(self.scene)
        self.ghost.add_to_scene(self.scene)
        for pellet in self.pellets:
            pellet.add_to_scene(self.scene)
            pellet.consumed.connect(self.on_pellet_consumed_)

        # Game text for score and completion messages
        self.score_text = GameText(text=f"Score: {self.score}", x=10, y=10, font_size=14, color=QtCore.Qt.blue)
        
        self.message = GameText(text="Default", x=100, y=150, font_size=24, color=QtCore.Qt.white)
        self.message.hide()  # Hide initially
        self.scene.addItem(self.score_text)
        self.scene.addItem(self.message)

        scene_width = self.grid.num_columns * (self.grid.cell_size + self.grid.border_thickness)
        scene_height = self.grid.num_rows * (self.grid.cell_size + self.grid.border_thickness)
        self.scene.setSceneRect(0, 0, scene_width, scene_height)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ghost.exploded.connect(self.on_ghost_exploded_)
        self.collision_timer = QtCore.QTimer()
        self.collision_timer.timeout.connect(self.check_collisions)

    def start_game(self):
        self.running_ = True
        self.collision_timer.start(50)

    def stop_game(self):
        self.collision_timer.stop()
        self.running_ = False

    def check_collisions(self):
        """Check for collisions between Pacman, Ghost, and Goals."""
        if self.pacman.ellipse_item.collidesWithItem(self.ghost.ellipse_item):
            self.collision.emit(0)
        for pellet in self.pellets:
            if pellet.available:
                if self.pacman.ellipse_item.collidesWithItem(pellet.ellipse_item):
                    pellet.consume()
                    self.collision.emit(1)
        if self.running_:
            self.ghost.chase_pacman(self.pacman.ellipse_item.pos())

    def show_message(self, msg):
        """Update the complete_text message and display it on the screen."""
        self.message.update_text(msg)  # Set the new message text
        self.message.setPos(
            (self.scene.width() - self.message.boundingRect().width()) / 2,
            (self.scene.height() - self.message.boundingRect().height()) / 2
        )  # Center the text in the scene
        self.message.show()  # Display the message

    @pyqtSlot(int)
    def on_pellet_consumed_(self, value):
        """Handle a pellet being consumed by updating the score and checking for level completion."""
        self.score += value
        self.score_text.update_text(f"Score: {self.score}")  # Update the score text
        if self.score >= self.score_threshold_:
            self.stop_game()
            self.ghost.explode()

    @pyqtSlot()
    def on_ghost_exploded_(self):
        self.complete.emit(self.difficulty)
