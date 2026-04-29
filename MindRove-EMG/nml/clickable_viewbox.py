from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg

class ClickableViewBox(pg.ViewBox):
    plotClicked = pyqtSignal(object)  # Signal to emit when the plot is clicked
    
    def mouseClickEvent(self, event):
        if event.button() == pg.QtCore.Qt.LeftButton:  # Only handle left-clicks
            self.plotClicked.emit(event)
