from PyQt5 import QtWidgets, QtCore, QtGui

class GameText(QtWidgets.QGraphicsTextItem):
    def __init__(self, text="", x=0, y=0, font_size=16, color=QtCore.Qt.white):
        """
        Initialize a text item with customizable text, position, font size, and color.
        
        Args:
            text (str): The text content to display.
            x (int): The x-coordinate for positioning the text.
            y (int): The y-coordinate for positioning the text.
            font_size (int): Size of the font to be used.
            color (QtGui.QColor): Color of the text.
        """
        super().__init__(text)  # Initialize QGraphicsTextItem with the given text
        self.setPos(x, y)
        self.setDefaultTextColor(color)
        
        # Set font properties
        font = QtGui.QFont("Arial", font_size)
        font.setBold(True)
        self.setFont(font)

    def update_text(self, new_text):
        """Update the displayed text."""
        self.setPlainText(new_text)

    def set_color(self, color):
        """Change the text color."""
        self.setDefaultTextColor(color)

    def show_text(self):
        """Show the text item on the scene."""
        self.show()

    def hide_text(self):
        """Hide the text item from the scene."""
        self.hide()

