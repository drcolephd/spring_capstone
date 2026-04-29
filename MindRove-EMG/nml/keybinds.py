from PyQt5 import QtCore
from mindrove.board_shim import MindroveConfigMode

def keybinds_boardshim(self, source, event):
    if event.type() == QtCore.QEvent.KeyPress:
        if event.key() == QtCore.Qt.Key_Space:
            print("Spacebar pressed: Sending 'beep' sync")
            self.board_shim.config_board(MindroveConfigMode.BEEP)
        elif event.key() == QtCore.Qt.Key_Return:
            print("Enter pressed: Sending 'boop' sync")
            self.board_shim.config_board(MindroveConfigMode.BOOP)
        elif event.key() == QtCore.Qt.Key_Escape:
            print("Escape pressed: exiting application")
            self.close()
        elif event.key() == QtCore.Qt.Key_Up:
            print("Up")
            self.board_shim.insert_marker(7)
        elif event.key() == QtCore.Qt.Key_Down:
            print("Down")
            self.board_shim.insert_marker(9)
        elif event.key() == QtCore.Qt.Key_Left:
            print("Left")
            self.board_shim.insert_marker(10)
        elif event.key() == QtCore.Qt.Key_Right:
            print("Right")
            self.board_shim.insert_marker(8)
    return super().eventFilter(source, event)

def keybinds_marker(self, source, event):
    if event.type() == QtCore.QEvent.KeyPress:
        if event.key() == QtCore.Qt.Key_Space:
            print("Spacebar pressed: Marker = 1")
            self.marker.emit(1)
        elif event.key() == QtCore.Qt.Key_Return:
            print("Enter pressed: Marker = 2")
            self.marker.emit(2)
        elif event.key() == QtCore.Qt.Key_Escape:
            print("Escape pressed: exiting application")
            self.close()
        elif event.key() == QtCore.Qt.Key_Up:
            print("Up")
            self.control.emit(7)
            self.marker.emit(7)
        elif event.key() == QtCore.Qt.Key_Down:
            print("Down")
            self.control.emit(9)
            self.marker.emit(9)
        elif event.key() == QtCore.Qt.Key_Left:
            print("Left")
            self.control.emit(10)
            self.marker.emit(10)
        elif event.key() == QtCore.Qt.Key_Right:
            print("Right")
            self.control.emit(8)
            self.marker.emit(8)
    return super().eventFilter(source, event)