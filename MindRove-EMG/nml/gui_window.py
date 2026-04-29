from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
from nml.local_paths import paths
import matplotlib.cm as cm
import os

class GuiWindow(QtWidgets.QWidget):
    closed = QtCore.pyqtSignal()

    palette = {'pen': [], 'color': []}
    layout: QtWidgets.QGridLayout = None
    _canvas: pg.GraphicsWindow = None
    _assets: str = paths['assets']
    _configs: str = paths['configurations']
    _pen_width: float = 2.0

    _icon_image_file: str = "cmu-scotty-scarf.png"
    _background_image_file: str = "cmu-tartan-wave-gray-crop-01.png"

    def __init__(self, parent=None, 
                 set_stylesheet: bool = True, 
                 set_layout: bool = True, 
                 set_palette: bool = True, 
                 pen_width: float = 2.0, 
                 icon_image_file: str = None, 
                 background_image_file: str = None):
        super().__init__(parent)
        self._pen_width = pen_width
        if icon_image_file is not None:
            self._icon_image_file = icon_image_file
        if background_image_file is not None:
            self._background_image_file = background_image_file
        self.setWindowIcon(QtGui.QIcon(os.path.join(self._assets, self._icon_image_file)))
        if set_stylesheet:
            self.setStyleSheet(GuiWindow.getBaseStyleSheet())
        if set_layout:
            self._canvas = pg.GraphicsWindow()
            self.layout = QtWidgets.QGridLayout(self)
            self.layout.setSpacing(10)
            self.layout.addWidget(self._canvas,0,0,5,5)
        if set_palette:
            self.setPalette()

    def handleParentClosing(self):
        self.close()

    def closeEvent(self, event):
        """Handle the close event to ensure cleanup."""
        event.accept()  # Accept the close event
        self.closed.emit()  # Emit closed event

    def set_modal(self):
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

    def setPalette(self, base_colors = None, num_shades: int = 20):
        """
        Generate gradients from dark to light for each base color in the array.
        
        Parameters:
        - base_palette: (Default = None) List of RGB tuples representing base colors.
        - num_shades: int (Default = 20) Number of shades in the gradient for each color.
        """
        if base_colors is None:
            base_colors = GuiWindow.generateBaseColors()
        self.palette = {'pen': [], 'color': [], 'warm': {'pen': [], 'color': []}, 'cool': {'pen': [], 'color': []}}
        for color in base_colors:
            # Convert the color to a darker and lighter shade
            dark_color = tuple(int(c * 0.3) for c in color)  # 30% of original for dark
            light_color = tuple(int(min(c * 1.5, 255)) for c in color)  # 150% or max 255 for light
            
            # Interpolate between dark and light color
            pen_gradient = [pg.mkPen(width=self._pen_width,color=tuple((np.array(dark_color) * (1 - i) + np.array(light_color) * i).astype(int))) 
                        for i in np.linspace(0, 1, num_shades)]
            self.palette['pen'].append(pen_gradient)
            gradient = [pg.mkColor(tuple((np.array(dark_color) * (1 - i) + np.array(light_color) * i).astype(int))) 
                        for i in np.linspace(0, 1, num_shades)]
            self.palette['color'].append(gradient)
        warm_colors = GuiWindow.generateBaseColors("warm")
        for color in warm_colors:
            # Convert the color to a darker and lighter shade
            dark_color = tuple(int(c * 0.3) for c in color)  # 30% of original for dark
            light_color = tuple(int(min(c * 1.5, 255)) for c in color)  # 150% or max 255 for light
            
            # Interpolate between dark and light color
            pen_gradient = [pg.mkPen(width=self._pen_width,color=tuple((np.array(dark_color) * (1 - i) + np.array(light_color) * i).astype(int))) 
                        for i in np.linspace(0, 1, num_shades)]
            self.palette['warm']['pen'].append(pen_gradient)
            gradient = [pg.mkColor(tuple((np.array(dark_color) * (1 - i) + np.array(light_color) * i).astype(int))) 
                        for i in np.linspace(0, 1, num_shades)]
            self.palette['warm']['color'].append(gradient)

        cool_colors = GuiWindow.generateBaseColors("warm")
        for color in cool_colors:
            # Convert the color to a darker and lighter shade
            dark_color = tuple(int(c * 0.3) for c in color)  # 30% of original for dark
            light_color = tuple(int(min(c * 1.5, 255)) for c in color)  # 150% or max 255 for light
            
            # Interpolate between dark and light color
            pen_gradient = [pg.mkPen(width=self._pen_width,color=tuple((np.array(dark_color) * (1 - i) + np.array(light_color) * i).astype(int))) 
                        for i in np.linspace(0, 1, num_shades)]
            self.palette['cool']['pen'].append(pen_gradient)
            gradient = [pg.mkColor(tuple((np.array(dark_color) * (1 - i) + np.array(light_color) * i).astype(int))) 
                        for i in np.linspace(0, 1, num_shades)]
            self.palette['cool']['color'].append(gradient)

    def move_to_top_left(self):
        """
        Move the window to the top-left corner of the primary screen.
        """
        desktop = QtWidgets.QApplication.desktop()
        screen_geometry = desktop.screenGeometry(0)  # Primary screen
        self.move(screen_geometry.left(), screen_geometry.top())

    @staticmethod
    def getBaseStyleSheet() -> str:
        return """
            QWidget {
                background-color: black; /* Main background */
            }

            QLabel, QLineEdit, QCheckBox, QPushButton, QSpinBox, QComboBox {
                background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent for readability */
                color: black; 
                border: 1px solid #A50021; /* CMU red for subtle border */
                font-size: 16px;
                padding: 4px;
                border-radius: 4px;
            }

            /* Hover effect for QPushButton */
            QPushButton:hover {
                background-color: #A50021; /* CMU red */
                color: #FFFFFF; /* White text on hover */
                border: 1px solid #4D4D4D; /* Darker border on hover */
            }

            QPushButton:disabled {
                background-color: #6C6C6C; /* CMU gray */
                color: #FFFFFF;
            }

            /* Hover effect for QLineEdit */
            QLineEdit:hover {
                border: 1px solid #6C6C6C; /* CMU gray border on hover */
                background-color: rgba(255, 255, 255, 0.9); /* Slightly more opaque background */
            }

            /* Hover effect for QCheckBox */
            QCheckBox:hover {
                color: #A50021; /* Change checkbox text color on hover to CMU red */
            }
            QCheckBox::indicator:hover {
                background-color: #A50021; /* Highlight checkbox indicator on hover */
                border: 1px solid #4D4D4D; /* Darker border */
            }

            /* Hover effect for QComboBox */
            QComboBox:hover {
                border: 1px solid #A50021; /* Highlight border on hover */
                background-color: rgba(255, 255, 255, 0.9); /* Slightly more opaque background */
            }

            /* Hover effect for QSpinBox */
            QSpinBox:hover {
                border: 1px solid #6C6C6C; /* CMU gray border on hover */
                background-color: rgba(255, 255, 255, 0.9); /* Slightly more opaque background */
            }
        """

    def _initialize_cmu_style(self, window_background_alpha: float = 0.5, widget_background_alpha: int = 255, window_rgba = None):
        """Use CMU theme colors to make the styling nicer."""
        CMU_RED = "#A6192E"
        CMU_GRAY = "#58585B"
        CMU_DARK_GRAY = "#2D2926"
        CMU_WHITE = "#FFFFFF"
        CMU_BLACK = "#000000"

        if window_rgba is None:
            window_background_color = f"rgba(255, 255, 255, {window_background_alpha})"
        else:
            window_background_color = window_rgba

        # Setting the stylesheet in Application's init method
        img_file = os.path.join(paths['assets'], self._background_image_file).replace('\\','/')
        self.setStyleSheet(f"""
            QWidget {{
                background-color: rgba(0,0,0,1);
            }}
                           
            QMainWindow {{
                background-image: url({img_file});
                background-repeat: no-repeat;
                background-position: center;
                background-attachment: fixed;
                background-color: {window_background_color}; 
            }}

            QLineEdit, QCheckBox, QPushButton {{
                background-color: rgba(255, 255, 255, {widget_background_alpha}); 
                color: {CMU_GRAY}; 
                border: 1px solid {CMU_RED}; 
                font-size: 14px;
                padding: 4px;
                border-radius: 4px;
            }}

            QLabel {{
                background-color: {CMU_BLACK}; 
                color: {CMU_WHITE}; 
                border: 1px solid {CMU_RED}; 
                font-size: 14px;
                font-weight: bold; 
                padding: 4px;
                border-radius: 4px;
            }}

            QComboBox QAbstractItemView {{
                background-color: #f0f0f0;
                border: 1px solid #dcdcdc;
            }}

            QComboBox QAbstractItemView::item {{
                padding: 5px;
                background-color: #ffffff;
            }}

            QComboBox QAbstractItemView::item:selected {{
                background-color: #007bff;
                color: white;
            }}

            QPushButton:hover {{
                background-color: {CMU_RED};
                color: {CMU_WHITE}; 
                border: 1px solid {CMU_DARK_GRAY}; 
            }}

            QPushButton:disabled {{
                background-color: {CMU_GRAY};
                color: {CMU_WHITE};
            }}

            QLineEdit:hover {{
                border: 1px solid {CMU_GRAY}; 
                background-color: rgba(255, 255, 255, 0.9); 
            }}

            QCheckBox:hover {{
                color: {CMU_RED}; 
            }}
            QCheckBox::indicator:hover {{
                background-color: {CMU_RED}; 
                border: 1px solid {CMU_DARK_GRAY};
            }}

            QComboBox {{
                background-color: #f0f0f0;  
                border: 1px solid #dcdcdc;  
                padding: 5px; 
                border-radius: 5px;  
            }}

            
            QComboBox:hover {{
                border: 1px solid {CMU_RED};
                background-color: rgba(255, 255, 255, 0.9);
            }}

            QDoubleSpinBox, QSpinBox {{
                color: {CMU_GRAY}; 
                background-color: rgba(255, 255, 255, 1.0); 
                border: 1px solid {CMU_GRAY};  
                padding: 5px;  
                border-radius: 5px;  
            }}


            QDoubleSpinBox::down-button, QSpinBox::down-button {{
                background-color: {CMU_RED};  
                border: none; 
                border-radius: 3px;  
            }}

            QDoubleSpinBox::up-button, QSpinBox::down-button {{
                background-color: {CMU_DARK_GRAY};  
                border: none;  
                border-radius: 3px; 
            }}

            QDoubleSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, 
            QSpinBox::down-button:hover, 
            QSpinBox::up-button:hover {{
                background-color: #cccccc;  
            }}

        """)

    @staticmethod
    def get_colormap_colors(colormap_name: str, num_colors: int) -> list[tuple[int, int, int]]:
        """
        Generate a list of RGB colors from a specified Matplotlib colormap.

        Args:
            colormap_name (str): Name of the colormap (e.g., "coolwarm", "cool", "autumn").
            num_colors (int): Number of colors to sample.

        Returns:
            list[tuple[int, int, int]]: List of RGB colors as (R, G, B) tuples (0-255).
        """
        cmap = cm.get_cmap(colormap_name, num_colors)  # Load colormap with num_colors
        return [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_colors)]  # Convert to 0-255 RGB

    @staticmethod
    def generateBaseColors(palette_type: str = "default") -> list[tuple[int, int, int]]:
        """
        Generates a palette of RGB colors based on the specified palette type.

        Args:
            palette_type (str, optional): The type of color palette to generate. 
                - "default": Returns the default 8-color palette.
                - "warm": Returns a palette of 16 warm colors (reds, oranges, yellows).
                - "cool": Returns a palette of 8 cool colors (blues, purples, teals).

        Returns:
            list[tuple[int, int, int]]: A list of RGB tuples, where each tuple contains 
            three integers (0-255) representing a color.

        Example:
            >>> GuiWindow.generateBaseColors()
            [(48, 64, 166), (70, 118, 238), (27, 207, 213), ...]

            >>> GuiWindow.generateBaseColors("warm")
            [(255, 76, 51), (255, 127, 76), (229, 51, 25), ...]

            >>> GuiWindow.generateBaseColors("cool")
            [(51, 102, 255), (76, 127, 255), (51, 51, 229), ...]
        """
        match palette_type.lower():
            case "warm":
                # 64 warm colors (reds, oranges, yellows)
                warm_colors = GuiWindow.get_colormap_colors("autumn", 64)
                return warm_colors

            case "cool":
                # 8 cool colors (blues, purples, teals)
                cool_colors = GuiWindow.get_colormap_colors("cool", 8)
                return cool_colors

            case _:
                # Default 8 colors (same as before)
                default_colors = [
                    (0.1900, 0.2518, 0.6522),
                    (0.2769, 0.4658, 0.9370),
                    (0.1080, 0.8127, 0.8363),
                    (0.3857, 0.9896, 0.4202),
                    (0.8207, 0.9143, 0.2063),
                    (0.9967, 0.6082, 0.1778),
                    (0.8568, 0.2250, 0.1276),
                    (0.6896, 0.1158, 0.0806)
                ]
                return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in default_colors]