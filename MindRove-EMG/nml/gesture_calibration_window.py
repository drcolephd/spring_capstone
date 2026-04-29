from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, QObject, Qt

from nml.gui_window import GuiWindow
from nml.gesture import MlpGestureModel, MlpFitter

if TYPE_CHECKING:
    from nml.processor import Processor


class _TrainWorker(QObject):
    """Runs MlpFitter.fit in a background thread and emits the result."""

    done = pyqtSignal(object)  # MlpGestureModel | None

    def __init__(self, samples: list[tuple[np.ndarray, int]]):
        super().__init__()
        self._samples = samples

    @pyqtSlot()
    def run(self):
        model = MlpFitter.fit(self._samples)
        self.done.emit(model)


class GestureCalibrationWindow(GuiWindow):
    """
    Gesture calibration UI.

    Workflow:
      1. Configure gesture labels (add / rename / remove rows).
      2. Click "Record" for a gesture, perform the motion, click "Stop"
         (or let it auto-stop when the target sample count is reached).
      3. Repeat for every gesture until all have enough samples.
      4. Click "Train Model" — training runs in a background thread.
      5. After training, live predictions appear in the Prediction field.

    Connects to Processor.emg_features to receive pre-computed feature vectors.
    Emits gesture_model_trained with the fitted MlpGestureModel when done.
    """

    gesture_model_trained = pyqtSignal(object)  # MlpGestureModel | None

    _IDLE = 0
    _COLLECTING = 1
    _TRAINING = 2
    _RUNNING = 3

    def __init__(self, processor: "Processor"):
        super().__init__(set_layout=False)
        self._processor = processor
        self._state = self._IDLE
        self._active_label: int = -1
        self._samples: list[tuple[np.ndarray, int]] = []
        self._gesture_entries: list[dict] = []
        self._model: MlpGestureModel | None = None

        self.setWindowTitle("Gesture Calibration")
        self.setGeometry(620, 180, 500, 560)
        self._initialize_cmu_style()

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)
        self.setLayout(root)

        # Status bar
        self._status_lbl = QtWidgets.QLabel("Status: Idle")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._status_lbl)

        # Target samples row
        tgt_row = QtWidgets.QHBoxLayout()
        tgt_row.addWidget(QtWidgets.QLabel("Target samples / gesture:"))
        self._target_spin = QtWidgets.QSpinBox()
        self._target_spin.setRange(16, 5000)
        self._target_spin.setValue(150)
        tgt_row.addWidget(self._target_spin)
        root.addLayout(tgt_row)

        # Gesture list
        self._gesture_group = QtWidgets.QGroupBox("Gestures")
        self._gesture_vbox = QtWidgets.QVBoxLayout(self._gesture_group)
        self._gesture_vbox.setSpacing(4)
        root.addWidget(self._gesture_group)

        # Buttons row
        btn_row = QtWidgets.QHBoxLayout()
        self._add_btn = QtWidgets.QPushButton("+ Add Gesture")
        self._train_btn = QtWidgets.QPushButton("Train Model")
        self._reset_btn = QtWidgets.QPushButton("Reset")
        self._save_btn = QtWidgets.QPushButton("Save Model")
        self._load_btn = QtWidgets.QPushButton("Load Model")
        self._add_btn.clicked.connect(self._add_gesture_row)
        self._train_btn.clicked.connect(self._train)
        self._reset_btn.clicked.connect(self._reset_all)
        self._save_btn.clicked.connect(self._save_model)
        self._load_btn.clicked.connect(self._load_model)
        self._save_btn.setEnabled(False)
        for w in (self._add_btn, self._train_btn, self._reset_btn,
                  self._save_btn, self._load_btn):
            btn_row.addWidget(w)
        root.addLayout(btn_row)

        # Prediction display
        self._pred_lbl = QtWidgets.QLabel("Prediction: ---")
        self._pred_lbl.setAlignment(Qt.AlignCenter)
        font = self._pred_lbl.font()
        font.setPointSize(14)
        self._pred_lbl.setFont(font)
        root.addWidget(self._pred_lbl)

        # Sample count summary
        self._summary_lbl = QtWidgets.QLabel("")
        self._summary_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._summary_lbl)

        # Default gestures
        self._add_gesture_row("Rest")
        self._add_gesture_row("Gesture 1")

        # Wire to processor
        processor.emg_features.connect(self._on_features)
        self.closed.connect(lambda: processor.emg_features.disconnect(self._on_features))

    # ── Gesture row management ────────────────────────────────────────────────

    def _add_gesture_row(self, name: str = ""):
        label_int = len(self._gesture_entries)
        if not name:
            name = f"Gesture {label_int}"

        row_w = QtWidgets.QWidget()
        row_l = QtWidgets.QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)

        label_edit = QtWidgets.QLineEdit(name)
        label_edit.setFixedWidth(130)
        count_lbl = QtWidgets.QLabel("0")
        count_lbl.setFixedWidth(50)
        count_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        record_btn = QtWidgets.QPushButton("Record")
        remove_btn = QtWidgets.QPushButton("×")
        remove_btn.setFixedWidth(28)

        row_l.addWidget(label_edit)
        row_l.addWidget(QtWidgets.QLabel("samples:"))
        row_l.addWidget(count_lbl)
        row_l.addWidget(record_btn)
        row_l.addWidget(remove_btn)

        entry = {
            "label_edit": label_edit,
            "count_lbl": count_lbl,
            "record_btn": record_btn,
            "remove_btn": remove_btn,
            "label_int": label_int,
            "widget": row_w,
        }
        self._gesture_entries.append(entry)
        record_btn.clicked.connect(lambda _checked, e=entry: self._toggle_record(e))
        remove_btn.clicked.connect(lambda _checked, e=entry: self._remove_gesture_row(e))
        self._gesture_vbox.addWidget(row_w)
        self._update_summary()

    def _remove_gesture_row(self, entry: dict) -> None:
        if self._state != self._IDLE:
            return
        self._gesture_entries.remove(entry)
        entry["widget"].setParent(None)

    # ── Recording ─────────────────────────────────────────────────────────────

    def _toggle_record(self, entry: dict) -> None:
        if self._state == self._COLLECTING and self._active_label == entry["label_int"]:
            self._stop_collecting(entry)
        elif self._state == self._IDLE:
            self._start_collecting(entry)

    def _start_collecting(self, entry: dict) -> None:
        self._state = self._COLLECTING
        self._active_label = entry["label_int"]
        entry["record_btn"].setText("Stop")
        self._status_lbl.setText(f"Status: Collecting  '{entry['label_edit'].text()}'")
        self._set_controls_enabled(False)
        entry["record_btn"].setEnabled(True)

    def _stop_collecting(self, entry: dict) -> None:
        self._state = self._IDLE
        self._active_label = -1
        entry["record_btn"].setText("Record")
        self._status_lbl.setText("Status: Idle")
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, en: bool) -> None:
        for e in self._gesture_entries:
            e["record_btn"].setEnabled(en)
            e["remove_btn"].setEnabled(en)
        self._add_btn.setEnabled(en)
        self._train_btn.setEnabled(en)
        self._reset_btn.setEnabled(en)

    # ── Feature slot ─────────────────────────────────────────────────────────

    @pyqtSlot(object)
    def _on_features(self, feat: np.ndarray) -> None:
        if self._state == self._COLLECTING:
            self._samples.append((feat.copy(), self._active_label))
            label_int = self._active_label
            count = sum(1 for s in self._samples if s[1] == label_int)
            for e in self._gesture_entries:
                if e["label_int"] == label_int:
                    e["count_lbl"].setText(str(count))
                    if count >= self._target_spin.value():
                        self._stop_collecting(e)
                    break
            self._update_summary()

        elif self._state == self._RUNNING and self._model is not None:
            label_int, confidence = self._model.predict(feat)
            name = self._label_name(label_int)
            self._pred_lbl.setText(f"Prediction: {name}  ({100 * confidence:.1f}%)")

    # ── Training ──────────────────────────────────────────────────────────────

    def _train(self) -> None:
        if len(self._samples) < 64:
            QtWidgets.QMessageBox.warning(
                self, "Insufficient Data",
                "Need at least 64 samples total (≥16 per gesture)."
            )
            return
        self._state = self._TRAINING
        self._status_lbl.setText("Status: Training…")
        self._set_controls_enabled(False)
        self._save_btn.setEnabled(False)

        self._thread = QThread()
        self._worker = _TrainWorker(list(self._samples))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_train_done)
        self._thread.start()

    @pyqtSlot(object)
    def _on_train_done(self, model: MlpGestureModel | None) -> None:
        self._thread.quit()
        self._thread.wait()

        if model is None:
            self._state = self._IDLE
            self._status_lbl.setText("Status: Training failed — check sample counts")
            self._set_controls_enabled(True)
            return

        self._model = model
        self._state = self._RUNNING
        n_cls = model.num_classes
        self._status_lbl.setText(
            f"Status: Running  ({n_cls} classes, "
            f"feat dim {model.feature_dim})"
        )
        self._set_controls_enabled(True)
        self._save_btn.setEnabled(True)
        self.gesture_model_trained.emit(model)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def _save_model(self) -> None:
        if self._model is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Gesture Model", "gesture_model.json", "JSON (*.json)"
        )
        if path:
            self._model.save(path)

    def _load_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Gesture Model", "", "JSON (*.json)"
        )
        if not path:
            return
        model = MlpGestureModel.load(path)
        if model is None:
            QtWidgets.QMessageBox.warning(self, "Load Failed", "Could not load model from file.")
            return
        self._model = model
        self._state = self._RUNNING
        self._status_lbl.setText(
            f"Status: Running  (loaded — {model.num_classes} classes, "
            f"feat dim {model.feature_dim})"
        )
        self._save_btn.setEnabled(True)
        self.gesture_model_trained.emit(model)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_all(self) -> None:
        if self._state == self._TRAINING:
            return
        self._samples.clear()
        self._state = self._IDLE
        self._model = None
        self._active_label = -1
        self._status_lbl.setText("Status: Idle")
        self._pred_lbl.setText("Prediction: ---")
        self._save_btn.setEnabled(False)
        for e in self._gesture_entries:
            e["count_lbl"].setText("0")
            e["record_btn"].setText("Record")
        self._set_controls_enabled(True)
        self._update_summary()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _label_name(self, label_int: int) -> str:
        for e in self._gesture_entries:
            if e["label_int"] == label_int:
                return e["label_edit"].text()
        return str(label_int)

    def _update_summary(self) -> None:
        counts = {e["label_int"]: 0 for e in self._gesture_entries}
        for _, lbl in self._samples:
            if lbl in counts:
                counts[lbl] += 1
        parts = [
            f"{self._label_name(lbl)}: {n}"
            for lbl, n in counts.items()
        ]
        self._summary_lbl.setText("  |  ".join(parts) if parts else "")
