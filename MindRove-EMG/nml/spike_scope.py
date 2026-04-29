from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtGui import QColor
from nptyping import NDArray
import pyqtgraph as pg
import numpy as np
import os
from nml.local_paths import paths
from nml.clickable_viewbox import ClickableViewBox
from nml.gui_window import GuiWindow
from nml.priors import electrode_angles, electrode_text_angles
from scipy.signal import find_peaks, sosfilt

class SpikeScope(GuiWindow):
    rates: pyqtSignal = pyqtSignal(int, object)
    recruitment_order: pyqtSignal = pyqtSignal(object)
    covariance_state_change: pyqtSignal = pyqtSignal(float)

    _update_timer: QtCore.QTimer | None = None
    _update_timer_period: int = 35 # milliseconds
    _update_ipt_buffer_samples: int | None = None # based on _update_timer_period
    _state: int = 0

    _ipt_precision_spinbox: QtWidgets.QDoubleSpinBox | None = None
    _ipt_threshold_spinbox: QtWidgets.QDoubleSpinBox | None = None
    _buffering_covariance_checkbox: QtWidgets.QCheckBox | None = None
    _buffering_spikes_checkbox: QtWidgets.QCheckBox | None = None
    _calibrating_rates_checkbox: QtWidgets.QCheckBox | None = None
    _source_scope_mode_checkbox: QtWidgets.QCheckBox | None = None
    _adaptive_state_checkbox: QtWidgets.QCheckBox | None = None
    _set_ipt_max_button: QtWidgets.QPushButton | None = None
    _state_indicator_label: QtWidgets.QLabel | None = None
    _state_spinbox: QtWidgets.QSpinBox | None = None
    _source_ipt_spinbox: QtWidgets.QSpinBox | None = None
    _rate_calibration_threshold_spinbox: QtWidgets.QDoubleSpinBox | None = None
    _save_button: QtWidgets.QPushButton | None = None
    _load_button: QtWidgets.QPushButton | None = None
    _verbose_checkbox: QtWidgets.QCheckBox | None = None

    _rescale_flag: bool = False
    _max_scalar: float = -1.0
    _rate_calibration_threshold: float = 15.0
    _source_ipt: int = 0
    _source_scope_mode: bool = False # False = Channel Threshold-Crossings | True = Source-Space Threshold-Crossings
    _calibrating_thresholds: bool = False
    _ipt_thresholds: NDArray = np.ones((64,1))
    _realtime_mask: NDArray | None = None
    _spikes = None
    _buffer_duration: float = 0.5
    _source_signal_expected_range: float = 4.0
    _alpha = 0.75
    _beta = 0.25
    _template_curve = None
    _template_buffer: NDArray | None = None
    _curve_items = []
    _cluster_color = []
    _raster_scatter_items = []
    _embeddings_scatter_items = []
    _electrode_scatter_items = []
    _electrode_text_items = []
    _source_scatter_item = None
    _peak_ts = [0.0]  # Stores timestamps of peaks for ISI calculation
    _isis = []
    _num_waveforms = 20
    _current_spike = 0
    _plot = {'scope': None, 'rates': None, 'embeddings': None, 'isi': None, 'ipt': None, 'raster': None }
    _icon_image_file = "SpikeScope.png"
    _sample_rate = 500
    _num_ipts: int | None = None
    _num_embeddings = 16
    _embeddings_count = 0
    _rates = np.zeros(8, dtype=np.float32)
    _rate_lines = []
    _calibrating_rates: bool = False
    _calibrated_ipts: set = set()
    _pre_peak_samples: int | None = None
    _post_peak_samples: int | None = None
    _electrode_colors = None
    _channel = 0
    _per_channel_threshold = np.full(8, 1500.0)
    _threshold = 1500.0
    _time_indices = None
    _y_offset_source_scope = None
    _xdata_source_scope = None 
    _xdata_spike_scope = None
    _xdata_isi_bar_edges = None
    _isi_min: float = 0.0 # milliseconds
    _isi_max: float = 150.0 # milliseconds
    _num_isi_bars = 60
    _max_isi_history: int = 256 # Once we get this many isi for a channel, we pop oldest and append to list instead of only appending.
    
    def __init__(self, app, spikes, colors, channel: int = 0, threshold: float = 5000.0, sample_rate: int = 500, pre_peak_samples: int = 13, post_peak_samples: int = 18, num_embeddings: int = 16, num_ipts: int = 16, alpha: float = 0.5, parent = None):
        super().__init__(parent, set_stylesheet=False)
        self.app = app
        self._spikes = spikes
        self._time_indices = np.arange(self._spikes._ipt_buffer_size)  # X-axis: sample indices
        self.recruitment_order.connect(self._spikes.on_recruitment_order_change)
        self._spikes.source_thresholds_changed.connect(self._on_individual_ipt_thresholds_changed)
        self._spikes.num_ipts_changed.connect(self._on_num_ipts_changed)
        self._spikes.source_spike.connect(self.on_source_signal)
        self._spikes.threshold.connect(self.on_channel_thresholds_changed)
        self._buffer_duration = self._spikes._ipt_buffer_size / sample_rate
        self._pre_peak_samples = pre_peak_samples
        self._post_peak_samples = post_peak_samples
        self._realtime_mask = SpikeScope.get_rates_mask(N=self._spikes._ipt_buffer_size, 
                                                        post_peak_samples=self._post_peak_samples)
        self._channel = channel
        self._threshold = threshold
        self.set_alpha(alpha)
        self.setPalette(colors, self._num_waveforms)
        self._electrode_colors = colors
        self._sample_rate = sample_rate
        self._num_ipts = num_ipts
        self._num_embeddings = num_embeddings
        sample_rate_ms = self._sample_rate / 1000.0
        self._xdata_isi_bar_edges = np.linspace(self._isi_min, self._isi_max, self._num_isi_bars+1)  # Edges of ISI bars, milliseconds
        self._xdata_spike_scope = np.linspace(-self._pre_peak_samples / sample_rate_ms, self._post_peak_samples / sample_rate_ms, self._pre_peak_samples + self._post_peak_samples + 1)
        self._xdata_source_scope, self._ydata_source_scope = SpikeScope.generate_source_plot_offsets(extension_factor = (self._pre_peak_samples + self._post_peak_samples + 1))
        self._rates = np.zeros(self._num_ipts)
        self._electrode_angles = electrode_angles
        self._original_positions = np.array([
            [np.cos(angle), np.sin(angle)] for angle in self._electrode_angles
        ])
        self._template_buffer = np.zeros((self._num_waveforms, 8*(self._pre_peak_samples + self._post_peak_samples + 1)))
        self._update_ipt_buffer_samples = int(self._sample_rate / (float(self._update_timer_period) / 1000.0))
        # Initialize layout and plots
        self.setWindowTitle("Spike Scope")
        self.setWindowIcon(QtGui.QIcon(os.path.join(paths['assets'], self._icon_image_file)))
        self.setGeometry(400, 150, 1200, 750)

        self._initialize_cmu_style(window_rgba="rgba(0,0,0,1)")
        self._initialize_plots()
        self._initialize_widgets()

        # Set layout
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(self.layout)
        # Timer for continuous update of raster plots
        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._update_graphics)
        self._update_timer.start(self._update_timer_period)

    @pyqtSlot(object)
    def on_channel_thresholds_changed(self, spike_thresholds: np.ndarray):
        self._per_channel_threshold = spike_thresholds
        self.set_title(self._channel, self._per_channel_threshold[self._channel])

    def set_alpha(self, alpha: float):
        self._alpha = alpha
        self._beta = 1 - alpha

    def set_title(self, channel: int = 0, threshold: float = 5000.0):
        if self._source_scope_mode:
            return
        self._plot['scope'].setTitle(f"Spikes: Channel-{channel} | Threshold: {threshold/1000.0:.1f} mV")
        new_channel = not (channel == self._channel)
        self._channel = channel
        self._threshold = threshold
        self._per_channel_threshold[channel] = threshold
        if new_channel:
            self.reset_snippets()
            self.reset_isi()

    @pyqtSlot(object)
    def _on_individual_ipt_thresholds_changed(self, thresholds: "np.ndarray"):
        print(thresholds)
        self._ipt_thresholds[:,:] = thresholds[:, np.newaxis]
        print("Set individual IPT thresholds.")

    @pyqtSlot(int)
    def _on_num_ipts_changed(self, new_count: int):
        self._source_ipt = min(self._source_ipt, new_count)
        self._source_ipt_spinbox.setValue(self._source_ipt)
        self._source_ipt_spinbox.setRange(0, new_count)

    @pyqtSlot(int, QColor)
    def on_cluster_color_assigned(self, clus: int, col: QColor):
        # self._raster_scatter_items[clus].setPen(pg.mkPen(color=col, width=0.5))
        col.setAlphaF(0.5)
        self._cluster_color[clus] = col
        self._rate_lines[clus].setPen(pg.mkPen(color=col, width=2.5))
        if self._source_scope_mode:
            if clus == self._source_ipt:
                for curve in self._curve_items:
                    curve.setPen(pg.mkPen(color=col, width=1))

    @pyqtSlot(bool)
    def toggle_threshold_calibration(self, state: bool):
        """
        Starts or stops the IPT threshold calibration.
        """
        if state:
            self._calibrating_thresholds = True
            self._calibrated_ipts = set()
            # Ensure we pass an integer to numpy.arange to satisfy typing/stub signatures
            self._calibration_order = np.arange(int(self._num_ipts), dtype=int)  # Reset order
        else:
            self.stop_threshold_calibration()

    def stop_threshold_calibration(self):
        """
        Stops the calibration and emits the final indexing order.
        """
        if self._calibrating_thresholds:
            self._calibrating_thresholds = False
            self._calibrating_rates_checkbox.setChecked(False)
            self.recruitment_order.emit(self._calibration_order)

    def _update_threshold_calibration(self):
        """
        Updates IPT rankings based on spike rate activity.
        """
        if not self._calibrating_thresholds:
            return
        
        # Track which IPTs have exceeded threshold
        active_ipts = np.where(self._rates >= self._rate_calibration_threshold)[0]
        new_ipts = [ipt for ipt in active_ipts if ipt not in self._calibrated_ipts]

        for ipt in new_ipts:
            # Move IPT to front of _calibration_order
            self._calibration_order = np.concatenate((
                [ipt], self._calibration_order[self._calibration_order != ipt]
            ))
            self._calibrated_ipts.add(ipt)  # Mark as reordered

        # Stop calibration if all IPTs have exceeded the threshold
        if len(self._calibrated_ipts) == self._num_ipts:
            self.stop_threshold_calibration()

    def reset_snippets(self):
        # Push the snippets "off" the view box (set out of yRange, basically):
        ydata = np.full(self._pre_peak_samples+self._post_peak_samples+1, 100.0) # type: ignore
        for i, curve in enumerate(self._curve_items):
            curve.setData(self._xdata_spike_scope, ydata)
            curve.setPen(self.palette['pen'][self._channel][i])
        self._current_spike = 0

    def reset_isi(self):
        if self._source_scope_mode:
            self._isi_bars.setOpts(height=np.zeros(self._num_isi_bars), brush=pg.mkBrush(self._cluster_color[self._source_ipt]))
        else:
            self._isi_bars.setOpts(height=np.zeros(self._num_isi_bars), brush=pg.mkBrush(self._electrode_colors[self._channel]))
        self._peak_ts = [0.0]    
        self._isis = []

    def _initialize_plots(self):
        self._initialize_scope_plot()
        self._initialize_embeddings_plot()
        self._initialize_isi_plot()
        self._initialize_rates_plot(self._num_ipts)
        self._initialize_ipt_histogram()
        self._initialize_raster_plot(self._num_ipts)

    def _initialize_widgets(self):
        widget_layout = QtWidgets.QGridLayout()
        label = QtWidgets.QLabel("IPTs Threshold")
        label.setMaximumHeight(50)
        widget_layout.addWidget(label,0,0,1,1)
        self._ipt_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self._ipt_threshold_spinbox.setSingleStep(0.0025)
        self._ipt_threshold_spinbox.setRange(0,1)
        self._ipt_threshold_spinbox.setDecimals(4) 
        self._ipt_threshold_spinbox.setValue(self._spikes._scalar)
        self._ipt_threshold_spinbox.valueChanged.connect(self._set_ipt_threshold)
        widget_layout.addWidget(self._ipt_threshold_spinbox,0,1,1,1)

        self._buffering_covariance_checkbox = QtWidgets.QCheckBox()
        self._buffering_covariance_checkbox.setChecked(self._spikes._collecting_covariance_buffer)
        self._buffering_covariance_checkbox.setText("Covariance")
        self._buffering_covariance_checkbox.clicked.connect(self.set_covariance_buffering_state)
        widget_layout.addWidget(self._buffering_covariance_checkbox,1,0,1,1)
    
        self._buffering_spikes_checkbox = QtWidgets.QCheckBox()
        self._buffering_spikes_checkbox.setChecked(self._spikes._collecting_spike_buffer)
        self._buffering_spikes_checkbox.setText("Calibrate")
        self._buffering_spikes_checkbox.clicked.connect(self.set_spike_buffering_state)
        widget_layout.addWidget(self._buffering_spikes_checkbox,1,1,1,1)

        self._set_ipt_max_button = QtWidgets.QPushButton()
        self._set_ipt_max_button.setText("Set IPT Maxima")
        self._set_ipt_max_button.clicked.connect(self._spikes.set_max_values)
        widget_layout.addWidget(self._set_ipt_max_button,2,0,1,1)

        self._source_scope_mode_checkbox = QtWidgets.QCheckBox()
        self._source_scope_mode_checkbox.setChecked(self._source_scope_mode)
        self._source_scope_mode_checkbox.setText("Source Mode")
        self._source_scope_mode_checkbox.clicked.connect(self.toggle_source_scope_mode)
        widget_layout.addWidget(self._source_scope_mode_checkbox,2,1,1,1)

        label = QtWidgets.QLabel("IPT Source")
        label.setMaximumHeight(50)
        widget_layout.addWidget(label,3,0,1,1)

        self._source_ipt_spinbox = QtWidgets.QSpinBox()
        self._source_ipt_spinbox.setValue(0)
        self._source_ipt_spinbox.setRange(0, self._spikes._num_ipts-1)
        widget_layout.addWidget(self._source_ipt_spinbox, 3, 1, 1, 1)
        self._source_ipt_spinbox.valueChanged.connect(self.set_source_ipt)
        self._source_ipt_spinbox.setEnabled(False)

        self._calibrating_rates_checkbox = QtWidgets.QCheckBox()
        self._calibrating_rates_checkbox.setChecked(False)
        self._calibrating_rates_checkbox.setText("Recruit")
        self._calibrating_rates_checkbox.clicked.connect(self.toggle_threshold_calibration)
        widget_layout.addWidget(self._calibrating_rates_checkbox,4,0,1,1)

        self._state_indicator_label = QtWidgets.QLabel("State 0 Not Set")
        self._state_indicator_label.setMaximumHeight(50)
        self._state_indicator_label.setStyleSheet("color: red;")
        widget_layout.addWidget(self._state_indicator_label,4,1,1,1)

        self._adaptive_state_checkbox = QtWidgets.QCheckBox()
        self._adaptive_state_checkbox.setChecked(self._spikes._enable_state_adaptation)
        self._adaptive_state_checkbox.setText("Adaptive")
        self._adaptive_state_checkbox.clicked.connect(self._handle_adaptive_state_change)
        widget_layout.addWidget(self._adaptive_state_checkbox,5,0,1,1) 

        self._state_spinbox = QtWidgets.QSpinBox()
        self._state_spinbox.setPrefix("State: ")
        self._state_spinbox.setValue(self._spikes._state)
        self._state_spinbox.setRange(0, self._spikes._max_states-1)
        widget_layout.addWidget(self._state_spinbox, 5, 1, 1, 1)
        self._state_spinbox.valueChanged.connect(self._handle_changing_states)

        label = QtWidgets.QLabel("Rate Threshold")
        label.setMaximumHeight(50)
        widget_layout.addWidget(label,6,0,1,1)
        self._rate_calibration_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self._rate_calibration_threshold_spinbox.setSingleStep(1.0)
        self._rate_calibration_threshold_spinbox.setRange(0,40)
        self._rate_calibration_threshold_spinbox.setDecimals(0) 
        self._rate_calibration_threshold_spinbox.setValue(self._rate_calibration_threshold)
        self._rate_calibration_threshold_spinbox.valueChanged.connect(self.set_rate_calibration_threshold)
        widget_layout.addWidget(self._rate_calibration_threshold_spinbox,6,1,1,1)

        label = QtWidgets.QLabel("IPT Precision")
        label.setMaximumHeight(50)
        widget_layout.addWidget(label,7,0,1,1)
        self._ipt_precision_spinbox = QtWidgets.QDoubleSpinBox()
        self._ipt_precision_spinbox.setSingleStep(0.001)
        self._ipt_precision_spinbox.setRange(0.001,0.25)
        self._ipt_precision_spinbox.setValue(0.100)
        self._ipt_precision_spinbox.setDecimals(3) 
        self._ipt_precision_spinbox.valueChanged.connect(self._handle_setting_precision)
        widget_layout.addWidget(self._ipt_precision_spinbox,7,1,1,1)

        self._save_button = QtWidgets.QPushButton()
        self._save_button.setText("Save")
        self._save_button.clicked.connect(self._spikes.save)
        widget_layout.addWidget(self._save_button,8,0,1,1)

        self._load_button = QtWidgets.QPushButton()
        self._load_button.setText("Load")
        self._load_button.clicked.connect(self._handle_loading_spikes_model)
        widget_layout.addWidget(self._load_button,8,1,1,1)

        self._verbose_checkbox = QtWidgets.QCheckBox()
        self._verbose_checkbox.setChecked(self._spikes._verbose)
        self._verbose_checkbox.setText("Verbose")
        self._verbose_checkbox.clicked.connect(self._handle_setting_verbose_state)
        widget_layout.addWidget(self._verbose_checkbox,9,0,1,1)

        self.layout.addLayout(widget_layout, 0, 5, 5, 1)

    @pyqtSlot(bool)
    def _handle_setting_verbose_state(self, verbose_state: bool):
        self._spikes.set_verbose(verbose_state)
        if verbose_state:
            print("Verbose Mode Enabled")
        else:
            print("Verbose Mode Off")

    @pyqtSlot(float)
    def _handle_setting_precision(self, new_precision: float):
        self._spikes.set_precision(new_precision)

    @pyqtSlot()
    def _handle_loading_spikes_model(self):
        self._spikes.load()
        self.set_state_status(self._spikes._state, self._spikes._has_muaps)
        self._ipt_threshold_spinbox.setValue(self._spikes._scalar)
        self._source_ipt_spinbox.setValue(self._spikes._source)
        self._ipt_precision_spinbox.setValue(self._spikes._precision)

    @pyqtSlot(int)
    def _handle_changing_states(self, new_state: int):
        self._state = new_state
        self._spikes.set_state(new_state)
        has_valid_filters = (self._spikes._P[new_state] is not None) and (self._spikes._muap_filters[new_state] is not None)
        self.set_state_status(new_state, has_valid_filters)

    def set_state_status(self, state_index: int, has_valid_filters: bool):
        if has_valid_filters:
            self._state_indicator_label.setText(f"State {state_index} Set")
            self._state_indicator_label.setStyleSheet("color: green;")
        else:
            self._state_indicator_label.setText(f"State {state_index} Not Set")
            self._state_indicator_label.setStyleSheet("color: red;")

    @pyqtSlot(bool)
    def _handle_adaptive_state_change(self, state: bool):
        self._spikes.set_adaptive(state)
        if state:
            self._state_spinbox.setEnabled(False) # Do not allow manual state toggle while in adaptive mode.
            self._state_spinbox.valueChanged.disconnect(self._handle_changing_states)
        else:
            self._state_spinbox.setEnabled(True)
            self._state_spinbox.valueChanged.connect(self._handle_changing_states)

    _template_brush: list = None
    _template_x: list = None
    _template_y: list = None

    def extract_template_positions(self):
        self._template_brush = []
        self._template_x = []
        self._template_y = []
        for ipt_index in range(self._spikes._num_ipts):
            template = self._spikes._muap_filters[self._spikes._state][ipt_index,:].reshape((self._spikes._extension_factor, 8))
            # template = 0.5 * template / np.max(np.abs(template))
            p2p_template = 0.35 * np.ptp(template / np.max(template), axis=0)
            template_x = np.dot(p2p_template,np.cos(self._electrode_angles))
            template_y = np.dot(p2p_template,-np.sin(self._electrode_angles))
            template_color = pg.mkColor(self._cluster_color[ipt_index])
            template_color.setAlphaF(0.75)
            self._template_brush.append(pg.mkBrush(template_color))
            self._template_x.append(template_x)
            self._template_y.append(template_y)

    @pyqtSlot(int)
    def set_source_ipt(self, ipt_index):
        prev_ipt = self._source_ipt
        self._source_ipt = ipt_index
        self._spikes.set_ipt_threshold_source(ipt_index)
        self._source_ipt_spinbox.setValue(ipt_index)

        if self._source_scope_mode:
            self._plot['scope'].setTitle(f"Spikes: Source-{self._source_ipt}")
            for curve in self._curve_items:
                curve.setPen(pg.mkPen(color=self._cluster_color[ipt_index], width=1))
                curve.setData(self._xdata_source_scope, np.full_like(self._xdata_source_scope,-100))
            self._template_buffer = np.zeros((self._num_waveforms, 8*(self._pre_peak_samples + self._post_peak_samples + 1)))
            self._template_curve.setData(self._xdata_source_scope, self._ydata_source_scope)
            self.reset_isi()
            self.refresh_ipt_histogram_edges()
            self._ipt_bars.setOpts(x=self._xdata_ipt_bar_edges[:-1], height=np.zeros_like(self._xdata_ipt_bar_edges[:-1]),brush=pg.mkBrush(color=self._cluster_color[ipt_index]))
            self._rate_lines[prev_ipt].setShadowPen(pg.mkPen(color="black", width=3, cosmetic=True))
            self._rate_lines[self._source_ipt].setShadowPen(pg.mkPen(color="white", width=3, cosmetic=True))
            self._source_scatter_item.setBrush(self._template_brush[ipt_index])
            self._source_scatter_item.setData([self._template_x[ipt_index]], [self._template_y[ipt_index]])
            print(f"Source IPT Thresholds: ({self._spikes._threshold[self._spikes._state][ipt_index][0]:.3f}, {self._spikes._limit[self._spikes._state][ipt_index][0]:.3f})")

    @pyqtSlot(float)
    def set_rate_calibration_threshold(self, new_rate_threshold: float):
        self._rate_calibration_threshold = new_rate_threshold

    @pyqtSlot(bool)
    def set_spike_buffering_state(self, state):
        self._spikes.set_spike_buffering_state(state)
        self._buffering_spikes_checkbox.setChecked(state)

    @pyqtSlot(bool)
    def toggle_source_scope_mode(self, state: bool):
        self._source_scope_mode = state
        if state:
            self._initialize_source_scope_plot()
            if self._source_scatter_item is None:
                self._source_scatter_item = pg.ScatterPlotItem(x=[0.0], y=[0.0], size=12, pen=pg.mkPen(color="white", width=1.5), brush=pg.mkBrush(self.palette['warm']['color'][0][-1]))
            self._plot['embeddings'].addItem(self._source_scatter_item)
            if self._template_brush is None:
                self.extract_template_positions()
            self.set_source_ipt(self._source_ipt)
            self._spikes.enable_source_mode(True)
            self._source_ipt_spinbox.setEnabled(True)
        else:
            self._initialize_scope_plot()
            self.set_scope_channel(self._channel)
            self._rate_lines[self._source_ipt].setShadowPen(pg.mkPen(color="black", width=3, cosmetic=True))
            self._source_ipt_spinbox.setEnabled(False)
            self._spikes.enable_source_mode(False)
            self._plot['embeddings'].removeItem(self._source_scatter_item)

    @pyqtSlot(int)
    def set_scope_channel(self, channel: int):
        self._channel = channel
        self._threshold = self._per_channel_threshold[channel]
        if not self._source_scope_mode:
            self.set_title(channel, self._threshold)
            self.reset_isi()

    @pyqtSlot(bool)
    def set_covariance_buffering_state(self, state: bool):
        self._spikes.set_covariance_buffering_state(state)
        self._buffering_covariance_checkbox.setChecked(state)

    @pyqtSlot(float)
    def _set_ipt_threshold(self, new_threshold: float):
        self._spikes.set_ipt_threshold_scalar(new_threshold)
        self.refresh_ipt_histogram_edges()
        self._ipt_bars.setOpts(x=self._xdata_ipt_bar_edges[:-1], height=np.zeros_like(self._xdata_ipt_bar_edges[:-1]))

    def _initialize_scope_plot(self):
        # Spike waveform plot
        if self._plot['scope'] is not None:
            if self._template_curve is not None:
                self._plot['scope'].removeItem(self._template_curve)
            self._canvas.removeItem(self._plot['scope'])
            del self._plot['scope']
            self._curve_items = []
            self._template_curve = None

        self._plot['scope'] = self._canvas.addPlot(
            title=f"Spikes: Channel-{self._channel} | Threshold: {self._threshold:.1f} mV", 
            labels={'left': 'Amplitude (mV)', 'bottom': 'Time-to-Peak (ms)'}, 
            viewbox=ClickableViewBox(),
            row=0,
            col=0,
            rowspan=3
        )
        self._plot['scope'].setYRange(-2, 2)
        self._plot['scope'].setXRange(self._xdata_spike_scope[0], self._xdata_spike_scope[-1])
        
        ydata = np.zeros(self._pre_peak_samples + self._post_peak_samples + 1)
        for i in range(self._num_waveforms):
            curve = pg.PlotCurveItem(pen=self.palette['pen'][self._channel][i])
            curve.setData(self._xdata_spike_scope, ydata)
            self._plot['scope'].addItem(curve)
            self._curve_items.append(curve)
    
    def _initialize_source_scope_plot(self):
        # Spike waveform plot
        if self._plot['scope'] is not None:
            self._canvas.removeItem(self._plot['scope'])
            del self._plot['scope']
            self._curve_items = []
            
        self._plot['scope'] = self._canvas.addPlot(
            title=f"Spikes: Source-{self._source_ipt}", 
            viewbox=ClickableViewBox(),
            row=0,
            col=0,
            rowspan=3
        )
        # y_check = np.where(~np.isinf(self._ydata_source_scope), self._ydata_source_scope, 0)
        # x_check = np.where(~np.isinf(self._xdata_source_scope), self._xdata_source_scope, 0)
        self._plot['scope'].setYRange(-3.125 * self._source_signal_expected_range, 3.125 * self._source_signal_expected_range)
        self._plot['scope'].setXRange(-3.125 * self._source_signal_expected_range, 3.125 * self._source_signal_expected_range)
        self._plot['scope'].showAxis('left', False)
        self._plot['scope'].showAxis('bottom', False)
        
        num_y_points = (self._pre_peak_samples + self._post_peak_samples + 2)*8
        ydata = np.full(num_y_points, np.inf).flatten()
        for i in range(self._num_waveforms):
            curve = pg.PlotCurveItem(
                pen=self.palette['pen'][self._channel][i],
                connect='finite',
                color=self._cluster_color[i]
            )
            curve.setData(self._xdata_source_scope, ydata)
            self._plot['scope'].addItem(curve)
            self._curve_items.append(curve)
        self._template_curve = pg.PlotCurveItem(pen=pg.mkPen(color="black", width=2.5), shadowpen=pg.mkPen(color="white", width=3.0), connect='finite')
        self._plot['scope'].addItem(self._template_curve)

    def _initialize_embeddings_plot(self):
        # Plot for displaying embeddings from recent spikes
        vb = ClickableViewBox()
        self._plot['embeddings'] = self._canvas.addPlot(
            row=3,
            col=0,
            rowspan=2,
            title="Spike Embeddings",
            viewBox=vb
        )
        # Scatter plot for spike embeddings
        for i in range(self._num_embeddings):
            scatter = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush("black"))
            self._plot['embeddings'].addItem(scatter)
            self._embeddings_scatter_items.append(scatter)
        self._electrode_text_angle = []
        _text_pos = []
        self._electrode_text_angle = electrode_text_angles
        # Add electrode markers and labels
        for k in range(8):
            x, y = np.cos(self._electrode_angles[k]), np.sin(self._electrode_angles[k])
            brush_color = self._electrode_colors[k]
            
            # ✅ Add scatter point for electrode
            electrode = pg.ScatterPlotItem(x=[x], y=[y], size=12, pen=None, brush=pg.mkBrush(brush_color))
            self._plot['embeddings'].addItem(electrode)
            self._electrode_scatter_items.append(electrode)

            # # ✅ Compute rotation angle using atan2, ensuring a consistent rotation direction
            # adjusted_x = np.sign(y) * x  # Flip x if y is negative for consistent angle calculation
            # rotation_angle_rad = np.arctan2(y, adjusted_x)  # Compute angle using adjusted x
            # rotation_angle_deg = np.degrees(rotation_angle_rad) + 90  # Rotate tangent to the circle

            text_item = pg.TextItem(f"Ch-{k}", anchor=(0.5, 0.5), color=brush_color)
            text_item.setPos(x * 1.2, y * 1.2)  # Slightly offset text from the scatter point
            text_item.setRotation(self._electrode_text_angle[k])  # Rotate text to be tangent

            self._plot['embeddings'].addItem(text_item)
            self._electrode_text_items.append(text_item)
            _text_pos.append((x * 1.2, y * 1.2))
        self._electrode_text_pos = np.array(_text_pos)
    
        # Adjust plot settings
        self._plot['embeddings'].setYRange(-1.25, 1.25)
        self._plot['embeddings'].setXRange(-1.25, 1.25)
        self._plot['embeddings'].showAxis('left', False)
        self._plot['embeddings'].showAxis('bottom', False)
        vb.plotClicked.connect(lambda event: self._on_embeddings_plot_clicked(event))

    def _on_embeddings_plot_clicked(self, event):
        if not self._source_scope_mode:
            return
        mouse_point = self._plot['embeddings'].vb.mapSceneToView(event.scenePos())
        x, y = mouse_point.x(), mouse_point.y()
        # Convert lists to numpy arrays for efficient calculations
        template_x = np.array(self._template_x)
        template_y = np.array(self._template_y)

        # Compute squared Euclidean distances to avoid unnecessary sqrt computation
        distances = (template_x - x) ** 2 + (template_y - y) ** 2

        # Find the index of the closest coordinate
        ipt_template = np.argmin(distances)
        self.set_source_ipt(ipt_template)

    def _initialize_isi_plot(self):
        # Histogram plot for ISI
        self._plot['isi'] = self._canvas.addPlot(
            row=0,
            col=1,
            rowspan=2, 
            title="ISI (ms)" 
        )
        
        self._plot['isi'].showAxis('left', True)
        self._isi_bars = pg.BarGraphItem(x=self._xdata_isi_bar_edges[:-1], 
                                         width=(self._isi_max - self._isi_min)/self._num_isi_bars, 
                                         height=np.zeros_like(self._xdata_isi_bar_edges[:-1]), 
                                         pen=pg.mkPen(color="black", width=1), 
                                         brush=pg.mkBrush(self._electrode_colors[self._channel]))
        self._plot['isi'].addItem(self._isi_bars)
        self._plot['isi'].setXRange(self._isi_min, self._isi_max)
        self._plot['isi'].setYRange(0, 50)

    def _initialize_rates_plot(self, n: int):
        vb = ClickableViewBox()
        self._plot['rates'] = self._canvas.addPlot(
            row=3,
            col=1,
            rowspan=2,
            title="IPT Rates",
            viewBox=vb
        )
        self._plot['rates'].showAxis('left',False)
        self._plot['rates'].showAxis('right',False)
        self._plot['rates'].showAxis('top',False)
        self._plot['rates'].showAxis('bottom',True)

        # Create individual lines for each rate count
        self._rate_lines = []
        for i in range(n):
            # Define x-data as the rate value for each line, starting from zero
            x_data = [0, 0]
            y_data = [i, i]  # Y-position based on IPT row
            self._cluster_color.append(self.palette['warm']['color'][i][-1])
            line = pg.PlotCurveItem(x=x_data, y=y_data, pen=pg.mkPen(color=self._cluster_color[i], width=2), shadowpen=pg.mkPen(color="black", width=3, cosmetic=True))
            self._plot['rates'].addItem(line)
            self._rate_lines.append(line)
        self._plot['rates'].setXRange(0, 30)  # Adjust for expected max rate
        self._plot['rates'].setYRange(-0.5, n-0.5)
        self._plot['rates'].setLabel('bottom', "spikes/sec")
        vb.plotClicked.connect(lambda event: self._on_rates_plot_clicked(event))

    def _on_rates_plot_clicked(self, event):
        if not self._source_scope_mode:
            return
        mouse_point = self._plot['rates'].vb.mapSceneToView(event.scenePos())
        y = min(max(round(mouse_point.y()), 0), self._num_ipts-1)
        # print(f"Selected IPT Source: {y}")
        self.set_source_ipt(y)

    def update_rate_lines(self):
        """Updates the rate bar lines for each cluster based on the rate counts."""
        for i, rate in enumerate(self._rates):
            x_data = [0, rate]
            y_data = [i, i]
            self._rate_lines[i].setData(x=x_data, y=y_data)
        self.rates.emit(self._rates.size, self._rates)

    _ipt_bars: pg.BarGraphItem = None
    def _initialize_ipt_histogram(self):
        # Histogram plot for ISI
        self._plot['ipt'] = self._canvas.addPlot(
            row=0,
            col=2,
            rowspan=2, 
            title="IPT (pulse power)" 
        )
        self.refresh_ipt_histogram_edges()
        self._plot['ipt'].showAxis('left', False)
        self._plot['ipt'].showAxis('right', True)
        self._ipt_bars = pg.BarGraphItem(x=self._xdata_ipt_bar_edges[:-1], 
                                         width=np.diff(self._xdata_ipt_bar_edges)[0], 
                                         height=np.zeros_like(self._xdata_ipt_bar_edges[:-1]), 
                                         pen=pg.mkPen(color="black", width=1), 
                                         brush=pg.mkBrush(self._cluster_color[0]))
        self._plot['ipt'].addItem(self._ipt_bars)
        self._plot['ipt'].setXRange(self._ipt_bar_min, self._ipt_bar_max)
        self._plot['ipt'].setYRange(0, 100)

    _ipt_bar_min: float = None
    _ipt_bar_max: float = None
    _ipt_val_history: int = 512
    _num_ipt_bars: int = 20
    _xdata_ipt_bar_edges: "np.ndarray" = None

    def refresh_ipt_histogram_edges(self):
        self._ipt_histogram_values = np.zeros(self._ipt_val_history)
        self._ipt_bar_min = self._spikes._threshold[self._spikes._state][self._spikes._source][0]
        self._ipt_bar_max = self._spikes._limit[self._spikes._state][self._spikes._source][0]
        print(f"IPT Histogram Limits (Source-{self._spikes._source}): [{self._ipt_bar_min:.3f}, {self._ipt_bar_max:.3f}]")
        self._xdata_ipt_bar_edges = np.linspace(self._ipt_bar_min, self._ipt_bar_max, self._num_ipt_bars+1)
        if self._ipt_bars is not None:
            self._ipt_bars.setOpts(x=self._xdata_ipt_bar_edges[:-1], height=np.zeros_like(self._xdata_ipt_bar_edges[:-1]), width=np.diff(self._xdata_ipt_bar_edges)[0])
        self._plot['ipt'].setXRange(self._ipt_bar_min, self._ipt_bar_max)

    def _initialize_raster_plot(self, n):
        self._plot['raster'] = self._canvas.addPlot(
            row=3,
            col=2,
            rowspan=2,
            title="Spike Raster"
        )
        self._plot['raster'].showAxis('left',False)
        self._plot['raster'].showAxis('right',True)
        self._plot['raster'].showAxis('top',False)
        self._plot['raster'].showAxis('bottom',True)
        self._raster_items = []
        for i in range(n):
            point = pg.ScatterPlotItem(size=4, symbol='arrow_up', pen=pg.mkPen(self._cluster_color[i], width=8))
            self._plot['raster'].addItem(point)
            self._raster_items.append(point)
        self._plot['raster'].setYRange(-1,self._num_ipts)
        self._plot['raster'].setXRange(0, self._spikes._ipt_buffer_size)
        self._plot['raster'].setLabel('bottom', "spike instants")

    def _update_graphics(self):
        self._update_raster()
        self.update_rate_lines()
        self._update_threshold_calibration()
        if self._spikes._enable_state_adaptation and (self._state != self._spikes._state):
            self._state = self._spikes._state
            self._state_spinbox.setValue(self._state)
            self._state_indicator_label.setText(f"State {self._state} Set")
            self.covariance_state_change.emit(self._state)

    def _update_raster(self):
        if not self._spikes._has_muaps:
            return
        for ipt in range(self._spikes._num_ipts):
            thresh = self._spikes._threshold[self._spikes._state][ipt]
            limit = self._spikes._limit[self._spikes._state][ipt]
            ipt_history = np.where((np.abs(self._spikes._ipt_normalized_buffer[ipt, :]) > thresh) & (np.abs(self._spikes._ipt_normalized_buffer[ipt, :]) > limit), 1, 0)
            # Compute rate counts for active segments
            self._rates[ipt] = np.dot(self._realtime_mask, ipt_history)
            # Find indices where IPT exceeds threshold
            active_indices = np.where(ipt_history > 0)[0]

            if len(active_indices) > 0:
                x_data = self._time_indices[active_indices]  # Time/sample indices where IPT is active
                y_data = np.full_like(x_data, ipt)     # Assign IPT index as Y value
            else:
                x_data, y_data = [], []  # No points, clear scatter plot

            # Update the scatter plot for this IPT
            self._raster_items[ipt].setData(x=x_data, y=y_data)
            if (ipt == self._spikes._source):
                values_above_threshold = self._spikes._ipt_normalized_buffer[ipt, -self._update_ipt_buffer_samples:].flatten()[(self._spikes._ipt_normalized_buffer[ipt, -self._update_ipt_buffer_samples:].flatten() > thresh) & (self._spikes._ipt_normalized_buffer[ipt, -self._update_ipt_buffer_samples:].flatten() < limit)].copy()
                num_values = values_above_threshold.size
                if num_values > 0:
                    self._ipt_histogram_values[:-num_values] = self._ipt_histogram_values[num_values:]

                    # Insert new values at the end
                    self._ipt_histogram_values[-num_values:] = values_above_threshold[-num_values:]
                    self._update_ipt_histogram(self._ipt_histogram_values)

    def _update_ipt_histogram(self, ipt_data: "np.ndarray"):
        # Add IPT data to histogram
        counts, _ = np.histogram(ipt_data, bins=self._xdata_ipt_bar_edges)
        self._ipt_bars.setOpts(height=counts)

    @pyqtSlot(float, float, int, int, float, int)
    def on_embedding_signal(self, embedding_x: float, embedding_y: float, sample: int, channel: int, orientation: float, cluster: int):
        """Slot function that updates the embedding scatter plot based on new embedding data."""
        self.update_embeddings_scatter(embedding_x, embedding_y, cluster, orientation)
        self._rotate_electrode_scatter_points(orientation)
        
    def update_embeddings_scatter(self, embedding_x: float, embedding_y: float, cluster: int, orientation: float = 0.0):
        # Set the position and color of the current embedding scatter item
        if self._source_scope_mode and self._source_ipt is not cluster:
            # If in source scope mode and no source IPT is set, do not update embeddings
            return
        cos_orientation = np.cos(orientation)
        sin_orientation = -np.sin(orientation)
        scatter_item = self._embeddings_scatter_items[self._embeddings_count]
        rotated_embedding = np.dot(np.array([embedding_x, embedding_y]), np.array([[cos_orientation, -sin_orientation], [sin_orientation, cos_orientation]]))
        scatter_item.setData([rotated_embedding[0]], [rotated_embedding[1]])

        # Get color of the corresponding rate line for this cluster
        col = pg.mkColor(self._cluster_color[cluster])
        col.setAlphaF(0.5)
        scatter_item.setBrush(pg.mkBrush(col))

        # Cycle to the next embedding scatter item
        self._embeddings_count = (self._embeddings_count + 1) % self._num_embeddings


    def _rotate_electrode_scatter_points(self, orientation):
        # Rotate the _electrode_scatter_items based on orientation
        cos_orientation = np.cos(orientation)
        sin_orientation = -np.sin(orientation)
        rotated_positions = np.dot(self._original_positions, np.array([[cos_orientation, -sin_orientation], [sin_orientation, cos_orientation]]))
        rotated_text_positions = np.dot(self._electrode_text_pos, np.array([[cos_orientation, -sin_orientation], [sin_orientation, cos_orientation]]))
        for k, scatter in enumerate(self._electrode_scatter_items):
            scatter.setData([rotated_positions[k, 0]], [rotated_positions[k, 1]])
            self._electrode_text_items[k].setPos(rotated_text_positions[k, 0], rotated_text_positions[k, 1])
            self._electrode_text_items[k].setRotation(self._electrode_text_angle[k] - np.degrees(orientation))  # Adjust text orientation

    @pyqtSlot(object, int, int)
    def on_spike_signal(self, waveform: np.ndarray, channel: int, sample: int):
        if not channel == self._channel:
            return
        if self._source_scope_mode:
            return
        # Update spike waveform plot
        self._curve_items[self._current_spike].setData(self._xdata_spike_scope, waveform / 1000.0)
        self._current_spike = (self._current_spike + 1) % self._num_waveforms

        # Update ISI distribution
        current_time = sample / self._sample_rate * 1000  # Convert to milliseconds
        isi = current_time - self._peak_ts[-1]
        self._isis.append(isi)
        if len(self._isis) > self._max_isi_history:
            self._isis.pop(0)  # Remove the oldest element if the list exceeds 256
        self._update_isi_histogram()
        self._peak_ts.append(current_time)

    @pyqtSlot(object, int, float)
    def on_source_signal(self, waveform: "np.array", sample: int, orientation: float = 0.0):
        if not self._source_scope_mode:
            return
        # print(waveform)
        waveform /= np.max(waveform)
        p2p_amplitudes = np.zeros((8,1),dtype=np.float32)
        p2p_amplitudes[:,:] = np.ptp(waveform.reshape((8,self._spikes._extension_factor)), axis=1)[:,np.newaxis] 
        embedding_x = np.dot((p2p_amplitudes).flatten(),np.cos(self._electrode_angles))*0.35
        embedding_y = np.dot((p2p_amplitudes).flatten(),np.sin(self._electrode_angles))*0.35
        self.update_embeddings_scatter(float(embedding_x), float(embedding_y), self._source_ipt, orientation)

        # Update spike waveform plot
        reindexed_waveform = self._ydata_source_scope.copy()
        # Get the valid indices where the data is not np.inf
        valid_indices = ~np.isinf(self._ydata_source_scope)

        # Ensure `waveform` is 1D and its shape matches the expected size
        self._template_buffer[self._current_spike, :] = np.asarray(waveform).reshape((1, 8 * (self._pre_peak_samples + self._post_peak_samples + 1)))

        # Assign waveform values only to the valid (non-inf) positions
        reindexed_waveform[valid_indices] = self._template_buffer[self._current_spike, :].flatten()
        self._curve_items[self._current_spike].setData(self._xdata_source_scope, reindexed_waveform + self._ydata_source_scope)
        ydata_source_template = self._ydata_source_scope.copy()
        ydata_source_template[valid_indices] = np.mean(self._template_buffer[:, :],axis=0).flatten()
        self._template_curve.setData(self._xdata_source_scope, ydata_source_template + self._ydata_source_scope)
        self._current_spike = (self._current_spike + 1) % self._num_waveforms

        # Update ISI distribution
        current_time = sample / self._sample_rate * 1000  # Convert to milliseconds
        isi = current_time - self._peak_ts[-1]
        self._isis.append(isi)
        if len(self._isis) > 256:
            self._isis.pop(0)  # Remove the oldest element if the list exceeds 256
        self._update_isi_histogram()
        self._peak_ts.append(current_time)

    def _update_isi_histogram(self):
        # Add ISI to histogram
        counts, _ = np.histogram([self._isis], bins=self._xdata_isi_bar_edges)
        self._isi_bars.setOpts(height=counts)

    @staticmethod
    def get_rates_mask(N: int, post_peak_samples: int, pulse_center_frac: float = 0.9, tau: float = 60.0, max_rate: float = 30) -> "np.ndarray":
        """
        Generates a Gaussian-shaped window centered in the later 2/3 of the buffer, normalized for event rate estimation.

        Parameters:
        - N (int): Number of samples in the buffer.
        - post_peak_samples (int): Number of samples after the peak.
        = pulse_center_frac (float): Should be between 0 and 1, location of center of Gaussian Pulse
        - tau (float): Standard deviation of the Gaussian (in samples).
        - max_rate (float): The maximum expected event rate (events per second).

        Returns:
        - np.ndarray: A normalized Gaussian-based rate mask.
        """
        # x = np.arange(N)  
        # center = int(pulse_center_frac * N)  # Center the Gaussian at 2/3 of the buffer

        # # Generate Gaussian function centered at 2/3 of the buffer
        # gauss_window = np.exp(-0.5 * ((x - center) / tau) ** 2)

        # # Compute the scaling factor based on `arange()` to match max_rate
        # scaling_factor = np.sum(np.exp(-0.5 * ((np.arange(0, N, post_peak_samples + 1) - center) / tau) ** 2))
        
        # # Normalize the Gaussian to maintain the expected max computed rate of 30/sec
        # window = gauss_window * max_rate / scaling_factor
        samples = np.exp(np.arange(0,N,post_peak_samples+1)/tau)
        window = np.exp(np.arange(N) / tau) 
        window /= (samples.sum() / max_rate)
        print(f"Rates-Mask: ({window.size},) | tau={tau:.2f}")
        return window

    
    @staticmethod
    def generate_source_plot_offsets(extension_factor=16):
        """
        Generates x_offset and y_offset vectors for plotting, incorporating electrode locations.
        
        Parameters:
            extension_factor (int): Number of samples per channel before adding a discontinuity.

        Returns:
            Tuple[np.ndarray, np.ndarray]: x_offset_padded, y_offset_padded (both shaped (136, 1)).
        """

        # Given angles around the unit circle for each channel (8 electrodes)
        angles = electrode_angles
        
        num_channels = len(angles)  # 8 channels
        num_samples = num_channels * extension_factor  # 128 total samples
        total_samples = num_samples + num_channels  # 136 total with discontinuities
        
        # Generate linearly increasing x-values within each channel
        x_base = np.linspace(-2.5, 2.5, extension_factor)  # Shape: (16,)
        
        # Create empty arrays for padded values
        x_offset_padded = np.full((total_samples, 1), np.inf, dtype=np.float64).flatten()
        y_offset_padded = np.full((total_samples, 1), np.inf, dtype=np.float64).flatten()

        # Fill the padded arrays
        insert_index = 0
        for i in range(num_channels):
            # Compute electrode-based offsets
            x_channel_offset = x_base + np.cos(angles[i])*10.0  # Shape: (16,)
            y_channel_offset = np.full(extension_factor, np.sin(angles[i])*10.0).flatten()  # Shape: (16,)

            # Assign the values into the padded arrays
            x_offset_padded[insert_index:insert_index + extension_factor] = x_channel_offset
            y_offset_padded[insert_index:insert_index + extension_factor] = y_channel_offset

            # Move the insert index, leaving a gap for np.inf
            insert_index += extension_factor + 1  # Move past the inserted inf

        return x_offset_padded.flatten(), y_offset_padded.flatten()