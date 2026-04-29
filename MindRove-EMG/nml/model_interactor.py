from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QFileDialog
from nml.gui_window import GuiWindow
from nml.feature_weights import model as default_model
from nml.local_paths import paths
from nml.clickable_viewbox import ClickableViewBox
from typing import Tuple
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib.pyplot as plt
import os

class ModelInteractor(GuiWindow):
    model_update = pyqtSignal(object)

    coefficients: "np.ndarray[np.float32]" = None
    decode_mode: int = 1

    _bar_plot: Tuple[Tuple[pg.PlotItem]] = tuple()
    _coefficient_bar_items = None
    _saved_coefficients: "np.ndarray[np.float32]" = None

    def __init__(self, model=default_model, parent=None):
        super().__init__(parent, set_layout=False)
        self.setWindowTitle("Model Interactor")
        self.setWindowIcon(QtGui.QIcon(os.path.join(self._assets, "ModelInteractorIcon.png")))
        
        # Store model attributes
        self.coefficients = model['coeff']
        self._saved_coefficients = np.copy(model['coeff'])
        self.decode_mode = model['mode']
        
        # Initialize layout elements
        self._init_canvas()

    def _init_canvas(self):
        # Create layout components
        self.layout = QtWidgets.QGridLayout(self)

        # Create a label for decode mode
        mode_label = QtWidgets.QLabel("Decode Mode:")
        self.layout.addWidget(mode_label, 0, 0)
        
        # Decode mode display
        if self.decode_mode == 1:
            s = "Angular Velocity"
        elif self.decode_mode == 2:
            s = "Angular Acceleration"
        else:
            s = f"{self.decode_mode}"
        mode_display = QtWidgets.QLabel(s)
        mode_display.setAlignment(QtCore.Qt.AlignLeft)
        self.layout.addWidget(mode_display, 0, 1)

        # Save button
        save_button = QtWidgets.QPushButton("Save")
        save_button.clicked.connect(self.save_coefficients)
        self.layout.addWidget(save_button, 0, 2, 1, 1)

        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.clicked.connect(self.reset_coefficients)
        self.layout.addWidget(reset_button, 0, 3, 1, 1)

        self._canvas = pg.GraphicsWindow()
        self.layout.addWidget(self._canvas, 1, 0, 4, 4)

        # Initialize the bar plot grid (9 rows x 2 columns)
        self._bar_plot = tuple(tuple(pg.PlotItem(viewBox=ClickableViewBox()) for _ in range(2)) for _ in range(9))

        # Configure each plot and connect clicks
        for i in range(9):
            for j in range(2):
                plot = self._bar_plot[i][j]
                plot.setYRange(-2 * self.coefficients[i,j], 2 * self.coefficients[i,j])
                plot.setXRange(-0.55, 0.55)
                plot.showAxis('bottom', False)
                plot.setTitle(f'coeff[{i},{j}]')
                vb = plot.getViewBox()
                vb.plotClicked.connect(lambda event, row=i, col=j: self._on_model_coefficient_viewbox_click(event, row, col))

                # Set initial bar heights based on coefficients
                if i == 0:
                    brush_color = QtGui.QColor("white")
                else:
                    brush_color = self.palette['color'][i-1][-1]
                bar = pg.BarGraphItem(x=[0], height=[self.coefficients[i, j]], width=0.5, brush=brush_color)
                plot.addItem(bar)
                self._canvas.addItem(plot, row=i, col=j)
        
        # Initialize bar heights for coefficients
        self._set_bar_heights()

    def _on_model_coefficient_viewbox_click(self, event, row, col):
        mouse_point = self._bar_plot[row][col].vb.mapSceneToView(event.scenePos())
        new_value = mouse_point.y()
        self.coefficients[row, col] = new_value
        self._set_bar_heights()  # Update the bar heights

    def save_coefficients(self):
        # Save the modified coefficients back to the model or write to file
        ModelInteractor.print_model(self.coefficients)
        self._saved_coefficients = self.coefficients
        self.model_update.emit(self.coefficients)

    def reset_coefficients(self):
        """
        Reset coefficients to values in _saved_model.
        """
        print("Reset coefficients.")
        self.coefficients = self._saved_coefficients
        self._set_bar_heights()

    @pyqtSlot(object)
    def on_coefficient_adaptation(self, updated_coefficients: "np.ndarray[np.float32]"):
        self.coefficients = updated_coefficients
        self._set_bar_heights()

    def _set_bar_heights(self):
        """
        Sets each bar height that represents a model coefficient to its corresponding value in self.coefficients.
        """
        for i in range(9):
            for j in range(2):
                bar = self._bar_plot[i][j].items[0]  # Assuming each plot has a single BarGraphItem
                bar.setOpts(height=[self.coefficients[i, j]])  # Update bar height

    @staticmethod
    def print_model(beta_pls: "np.ndarray[np.float32]", file_path: str = None, default_name: str = None, file_comment: str = "#Angular Velocity Model\n") -> str:
        print("\nmodel={")
        print("\t'mode': 1, ")
        print("\t'coeff': np.array([")
        for row in beta_pls:
            print(f"\t[{', '.join(f'{x:.6e}' for x in row)}],")
        print("\t])")
        print("}")

        if file_path is None:
            # Prompt for save path
            if default_name is None:
                def_path = paths['models']
            else:
                def_path = os.path.join(paths['models'], default_name)
            file_path, _ = QFileDialog.getSaveFileName(
                None, "Save Model Coefficients", def_path, "CSV Files (*.csv);;All Files (*)"
            )
        # Save to CSV if a file path was provided
        if file_path:
            np.savetxt(file_path, beta_pls, delimiter=",", header="x_model,y_model", comments=file_comment, fmt="%.6e")
            print(f"Model coefficients saved to {file_path}")
            return file_path
        else:
            print("Save canceled.")
            return None

    @staticmethod
    def perform_pls_error_regression(buffer, n: int = 4) -> "np.ndarray[np.float32]":
        """
        Uses historical data buffer to perform PLS regression on envelope data and errors.
        """
        # Ensure there is enough data to perform regression
        if len(buffer) < 10:
            print("Insufficient data in buffer for PLS regression.")
            return
        
        # Extract data from the buffer
        control_data = np.array([[*entry["envelope"], entry["dx"], entry["dy"]] for entry in buffer])
        state_data = np.array([[entry["x_error"], entry["y_error"]] for entry in buffer])

        # Perform PLS regression on historical data
        pls = PLSRegression(n_components=n)
        pls.fit(control_data, state_data)

        # Store the updated coefficients (may add these to base model)
        error_model = np.vstack((pls.intercept_.T, pls.coef_.T))
        ModelInteractor.print_model(error_model, file_comment="#Angular Velocity Error Model\n")
        return error_model
    
    @staticmethod
    def perform_pls_regression(buffer, n: int = 4, def_name: str = None):
        """
        Uses historical data buffer to perform PLS regression on envelope data and errors.
        """
        # Ensure there is enough data to perform regression
        if len(buffer) < 10:
            print("Insufficient data in buffer for PLS regression.")
            return
        
        # Extract data from the buffer
        control_data = np.array([[*entry["sample"]] for entry in buffer])
        state_data = np.array([[entry["x"], entry["y"]] for entry in buffer])

        # Apply a low-pass filter to the 'x' and 'y' vectors
        sos_lpf, filter_state_lpf = ModelInteractor.initialize_lowpass_filter(
            sample_rate=50, filter_cutoff=3, filter_order=3, n_channels=2
        )
        state_data_smoothed = np.zeros_like(state_data)
        for i in range(2):  # Loop over 'x' and 'y' channels
            state_data_smoothed[:, i], filter_state_lpf[i] = sosfilt(sos_lpf, state_data[:, i], zi=filter_state_lpf[i] * state_data[0, i])

        # Perform PLS regression on historical data
        pls = PLSRegression(n_components=n)
        pls.fit(control_data, state_data)

        # Store the updated coefficients (may add these to base model)
        model = np.vstack((pls.intercept_.T, pls.coef_.T))
        state_estimate = control_data @ pls.coef_.T + pls.intercept_.T
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(state_data[:, 0], label="Measured", color="k")
        axes[0].plot(state_estimate[:, 0], label="Estimated", color="b", linestyle=":")
        axes[0].set_title("Horizontal Angular Velocity")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Angular Velocity (°/s)")

        axes[1].plot(state_data[:, 1], label="Measured", color="k")
        axes[1].plot(state_estimate[:, 1], label="Estimated", color="b", linestyle=":")
        axes[1].set_title("Vertical Angular Velocity")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Angular Velocity (°/s)")

        plt.legend()
        plt.tight_layout()

        model_file_path = ModelInteractor.print_model(model, file_comment="#Angular Velocity Model\n", default_name=def_name)
        if model_file_path:
            plot_file_path = model_file_path.replace(".csv", "_plot.png")
            plt.savefig(plot_file_path)
            print(f"Plot saved to {plot_file_path}")
            plt.show()
        else:
            plt.show()
            print("Plot not saved as model save was canceled.")
        return model, model_file_path
    
    @staticmethod
    def initialize_lowpass_filter(sample_rate: float, filter_cutoff: float, n_channels: int, filter_order: int = 3):
        nyquist = 0.5 * sample_rate
        cutoff_lpf = filter_cutoff / nyquist
        sos_lpf = butter(filter_order, cutoff_lpf, btype='low', output='sos')
        filter_state_lpf = np.array([sosfilt_zi(sos_lpf) for _ in range(n_channels)]) 
        return sos_lpf, filter_state_lpf
