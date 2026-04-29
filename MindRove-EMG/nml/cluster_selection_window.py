from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
from PyQt5.QtGui import QColor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from nml.local_paths import paths
import os

class ClusterSelectionWindow(QtWidgets.QDialog):
    # Signal to emit each cluster_id with its assigned color
    cluster_color_assigned = pyqtSignal(int, QColor)
    
    _checkboxes = []

    def __init__(self, cluster_templates, reduced_spikes, cluster_labels, k_clusters, parent=None):
        super().__init__(parent)
        self.cluster_templates = cluster_templates
        self.reduced_spikes = reduced_spikes
        self.cluster_labels = cluster_labels
        self.k_clusters = k_clusters
        self.selected_clusters = set()

        # Set up the layout and UI elements
        self.setWindowTitle("Cluster Selection")
        self.setModal(True)
        self.setWindowIcon(QtGui.QIcon(os.path.join(paths['assets'], "ClusteringWindowIcon.png")))
        self.setGeometry(150, 150, 1200, 850)
        main_layout = QtWidgets.QVBoxLayout()

        # Add a scrollable area for the template waveforms
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_content.setStyleSheet(self._get_stylesheet())
        scroll_layout = QtWidgets.QGridLayout(scroll_content)

        # Populate templates in a grid with checkboxes
        self.template_plots = []
        self.scatter_points = []
        colors = [QColor.fromHsvF(i / k_clusters, 1, 1).toRgb() for i in range(k_clusters)]
        _, cols = (self.k_clusters + 1) // 4, 4  # Adjust grid dimensions based on k_clusters
        for i in range(k_clusters):
            checkbox = QtWidgets.QCheckBox(f"Cluster {i}")
            checkbox.stateChanged.connect(lambda state, i=i: self._toggle_cluster(state, i))
            scroll_layout.addWidget(checkbox, np.floor(i // cols), (i % cols) * 2)

            # Reshape the cluster template to have 8 rows (channels) and the number of samples per channel
            cluster_template = cluster_templates[i].reshape(8, -1)  # Shape: (8, samples_per_channel)

            plot_widget = pg.PlotWidget()
            for channel_idx, channel_data in enumerate(cluster_template):
                plot_widget.plot(
                    np.arange(channel_data.size),  # X-axis values (sample indices)
                    channel_data + 2*channel_idx,  # Y-axis values for each channel
                    pen=pg.mkPen(color=colors[i])  
                )
            plot_widget.showAxis('left', False)
            plot_widget.showAxis('bottom', False)
            scroll_layout.addWidget(plot_widget, np.floor(i // cols), (i % cols) * 2 + 1)
            self.template_plots.append(plot_widget)
            self._checkboxes.append(checkbox)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # Scatter plot for the top 2 principal components
        self.scatter_plot = pg.PlotWidget(title="PCA Scatter Plot")
        for cluster_id in range(k_clusters):
            cluster_data = reduced_spikes[cluster_labels == cluster_id]
            scatter = pg.ScatterPlotItem(
                x=cluster_data[:, 0], y=cluster_data[:, 1],
                pen=pg.mkPen(None), brush=colors[cluster_id], size=8
            )
            self.scatter_plot.addItem(scatter)
            self.scatter_points.append(scatter)  # Store reference to update later
        self.scatter_plot.showAxis('left', False)
        self.scatter_plot.showAxis('bottom', False)
        main_layout.addWidget(self.scatter_plot)

        # Add confirm button
        confirm_button = QtWidgets.QPushButton("Confirm Selection")
        confirm_button.clicked.connect(self.confirm_selection)
        main_layout.addWidget(confirm_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)
        self.setStyleSheet(self._get_stylesheet())

    def toggle_all_clusters(self):
        for i in range(self.k_clusters):
            if self._checkboxes[i].isChecked():
                self._toggle_cluster(Qt.Unchecked, i)
                self._checkboxes[i].setCheckState(Qt.Unchecked)
            else:
                self._toggle_cluster(Qt.Checked, i)
                self._checkboxes[i].setCheckState(Qt.Checked)

    def _toggle_cluster(self, state, cluster_id):
        # Update selected clusters
        if state == Qt.Checked:
            self.selected_clusters.add(cluster_id)
        else:
            self.selected_clusters.discard(cluster_id)

        # Update edge color of scatter points for the selected cluster
        pen_color = pg.mkPen('yellow', width=2) if cluster_id in self.selected_clusters else None
        brush_color = self.template_plots[cluster_id].plotItem.listDataItems()[0].opts['pen'].color() if cluster_id in self.selected_clusters else QColor("black")
        
        self.scatter_points[cluster_id].setPen(pen_color)
        self.cluster_color_assigned.emit(cluster_id, brush_color)

    def confirm_selection(self):
        for cluster_id in self.selected_clusters:
            color = self.template_plots[cluster_id].plotItem.listDataItems()[0].opts['pen'].color()
            self.cluster_color_assigned.emit(cluster_id, color)
        self.accept()  # Close the window and signal that selection is complete

    def _get_stylesheet(self):
        return """
            QWidget {
                background-color: #000000;
            }

            QDialog {
                background-color: #000000;
                color: #e0e0e0;
                font-family: Arial;
                font-size: 12pt;
            }

            QScrollArea {
                background-color: #000000;
                border: 1px solid #555555;
            }

            QCheckBox {
                color: #e0e0e0;
                font-weight: bold;
                padding: 5px;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }

            QCheckBox::indicator:checked {
                background-color: #5a9bd4;
                border: 1px solid #5a9bd4;
            }

            QPushButton {
                background-color: #5a9bd4;
                color: white;
                border-radius: 5px;
                padding: 8px 16px;
            }

            QPushButton:hover {
                background-color: #488acb;
            }

            QPushButton:pressed {
                background-color: #3a6c9d;
            }
        """