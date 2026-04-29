from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import numpy as np
from typing import List, Tuple
from nml.gui_window import GuiWindow
import os

class SerialPlotWindow(GuiWindow):
    _n: int = 2
    _buffer: "np.ndarray[np.float32]" = None
    _buffer_size: int = 300
    _curves: List[pg.PlotCurveItem] = []
    _plot: pg.PlotItem = None
    _scatter: pg.PlotItem = None
    _selected: Tuple[int, int] = (0, 1)
    _counter: int = 0
    _xy: pg.ScatterPlotItem = None
    _xy_marker_size: float = 16.0
    # _xrange: Tuple[float, float] = [4.50, 3.70]
    # _yrange: Tuple[float, float] = [3.42, 2.65]
    _xrange: Tuple[float, float] = [2.515625, 3.34375]
    _yrange: Tuple[float, float] = [3.5625, 4.375]
    _curve_xdata: "np.ndarray[np.float32]" = None
    _button_state: bool = False
    _primary_color = None
    _secondary_color = None

    def __init__(self, n_channels: int = 2, buffer_size: int = 300, selected: tuple = (0, 1), marker_size: float = 16.0, refresh_rate: float = 30.0, parent=None):
        super().__init__(parent, pen_width=4.0)
        self._buffer_size = buffer_size
        self._xy_marker_size = marker_size
        self._buffer = np.zeros((n_channels, buffer_size), dtype=np.float32)
        self._curve_xdata = np.linspace(-self._buffer_size / refresh_rate, 0, self._buffer_size)
        self._initialize_plots(selected)
        self.set_n_channels(n_channels, selected)
        self.setWindowIcon(QtGui.QIcon(os.path.join(self._assets, "SerialCrossHairIcon.png")))
        self.setWindowTitle("Analog Serial Device Monitor")

    def set_scatter_limits(self, x_lim: Tuple[float, float], y_lim: Tuple[float, float]):
        """
        Update the x- and y-limits of the scatter plot.
        """
        self._xrange = x_lim
        self._yrange = y_lim

    def set_n_channels(self, n: int, selected: Tuple[int, int] = (0, 1)):
        """
        Update the number of channels.
        """
        self._n = n
        self._selected = selected
        self._counter = 0
        self._scatter.setTitle(f"A{selected[0]} vs A{selected[1]}")
        for curve in self._curves:
            del curve
        self._curves = []
        for i in range(self._n):
            curve = pg.PlotCurveItem(pen=self.palette['pen'][i % len(self.palette['pen'])][-1])  # Use palette colors
            self._plot.addItem(curve)
            self._curves.append(curve)

    def update(self, sample: "np.ndarray[np.float32]", button_state: bool):
        """
        Updates the time series plots and scatter plot based on incoming data.

        Parameters:
        - sample: A numpy array of shape (n_channels,) representing the new sample for each channel.
        - button_state: Bool which is True when button is asserted and False when button is deasserted.
        """
        # Shift buffer and add new data for each channel
        self._buffer = np.roll(self._buffer, -1, axis=1)
        self._buffer[:, -1] = sample

        # Update the line plots for each channel
        for i, curve in enumerate(self._curves):
            curve.setData(self._curve_xdata, self._buffer[i] + 2.5*i - 2.5)

        # Update the scatter plot with selected channel values
        x_data = SerialPlotWindow.remap(self._buffer[self._selected[0], -1], self._xrange)
        y_data = SerialPlotWindow.remap(self._buffer[self._selected[1], -1], self._yrange)
        self._xy.setData([x_data], [y_data])
        if self._button_state and not button_state:
            self._button_state = False
            self._xy.setPen(self._secondary_color)
        elif not self._button_state and button_state:
            self._button_state = True
            self._xy.setPen(self._primary_color)

        # Increment the sample counter
        self._counter = (self._counter + 1) % self._buffer_size

    @pyqtSlot(object, bool)
    def update_plot(self, data: "np.ndarray[np.float32]", button_state: bool):
        """Update the plots with new data from the Teensy."""
        self.update(data, button_state)

    def _initialize_plots(self, selected):
        """
        Creates two plots on the canvas: one for the time-series lines and other for the scatter plot.
        """
        # Create PlotCurveItems for each channel in the line plot
        self._plot = self._canvas.addPlot(title="Analog Streams", 
                                          labels={
                                              'bottom': 'Relative Sample Time (s)'
                                          })
        self._plot.showGrid(x=True, y=True)
        self._plot.showAxis('left', False)
        self._plot.showAxis('bottom', True)
        self._plot.setXRange(float(self._curve_xdata[0]), float(self._curve_xdata[-1]))
        self._plot.setYRange(-2.5, 2.5*self._n)
        
        # Initialize scatter plot for the selected channels
        self._scatter = self._canvas.addPlot(title=f"A{selected[0]} vs A{selected[1]}",
                                             labels={
                                                 'right': 'Normalized Vertical', 
                                                 'bottom': 'Normalized Horizontal'
                                             })
        self._scatter.setXRange(-1.0, 1.0)
        self._scatter.setYRange(-1.0, 1.0)
        self._scatter.showGrid(x=True, y=True)
        self._scatter.showAxis('left', False)
        self._scatter.showAxis('right', True)
        self._primary_color = self.palette['pen'][-2][-1]
        self._secondary_color = self.palette['pen'][1][-1]
        self._xy = pg.ScatterPlotItem(size=self._xy_marker_size, 
                                      pen=self._secondary_color, 
                                      brush=pg.mkBrush("white"))
        self._scatter.addItem(self._xy)

    @staticmethod
    def remap(data: float, data_lims: Tuple[float, float]) -> float:
        """
        Returns data remapped between -1.0 and 1.0 based on data limits.
        """
        data_c = (data_lims[0] + data_lims[1])/2
        data_r = (data_lims[1] - data_lims[0])/2
        return (data - data_c) / data_r