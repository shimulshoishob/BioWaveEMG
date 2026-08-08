import csv
import sys
import os
import time
import traceback
import numpy as np
import joblib

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from rf_features import extract_window_features, build_windows_from_sequence, feature_names

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR).lower() == "code":
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class ResultsDialog(QDialog):
    def __init__(self, summary_text, report_text, cm, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Random Forest Training Results")
        self.resize(980, 720)

        layout = QVBoxLayout(self)

        lbl_summary = QLabel(summary_text)
        lbl_summary.setWordWrap(True)
        lbl_summary.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_summary)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(report_text)
        txt.setStyleSheet("QTextEdit { font-family: Consolas; font-size: 12px; }")
        layout.addWidget(txt)

        lbl_cm = QLabel("Confusion Matrix")
        lbl_cm.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_cm)

        table = QTableWidget(cm.shape[0], cm.shape[1])
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setHorizontalHeaderLabels(class_names)
        table.setVerticalHeaderLabels(class_names)
        table.setStyleSheet("QTableWidget { font-family: Consolas; font-size: 12px; }")

        max_val = max(1, int(np.max(cm)))
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val = int(cm[r, c])
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                # Green intensity for higher counts.
                intensity = int(255 - (180 * (val / max_val)))
                item.setBackground(QColor(intensity, 255, intensity))
                table.setItem(r, c, item)

        layout.addWidget(table)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class RFTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Random Forest Trainer")
        self.resize(900, 680)

        if not HAS_SKLEARN:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "scikit-learn is not installed.\nInstall with:\npy -3 -m pip install scikit-learn",
            )

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        form = QFormLayout()
        main_layout.addLayout(form)

        self.input_csv = QLineEdit(os.path.join(DATASET_DIR, "emg_data_4ch_RAW.csv"))
        btn_csv = QPushButton("Browse")
        btn_csv.clicked.connect(self.browse_csv)
        row_csv = QHBoxLayout()
        row_csv.addWidget(self.input_csv)
        row_csv.addWidget(btn_csv)
        w_csv = QWidget()
        w_csv.setLayout(row_csv)
        form.addRow("Raw CSV:", w_csv)

        self.input_model = QLineEdit(os.path.join(TRAINED_MODEL_DIR, "rf_realtime_model.joblib"))
        btn_model = QPushButton("Browse")
        btn_model.clicked.connect(self.browse_model)
        row_model = QHBoxLayout()
        row_model.addWidget(self.input_model)
        row_model.addWidget(btn_model)
        w_model = QWidget()
        w_model.setLayout(row_model)
        form.addRow("Output Model:", w_model)

        self.spin_sr = QSpinBox()
        self.spin_sr.setRange(100, 5000)
        self.spin_sr.setValue(500)
        form.addRow("Sample Rate (Hz):", self.spin_sr)

        self.spin_win = QSpinBox()
        self.spin_win.setRange(20, 2000)
        self.spin_win.setValue(200)
        form.addRow("Window (ms):", self.spin_win)

        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(10, 1000)
        self.spin_stride.setValue(50)
        form.addRow("Stride (ms):", self.spin_stride)

        self.spin_estimators = QSpinBox()
        self.spin_estimators.setRange(50, 2000)
        self.spin_estimators.setValue(400)
        form.addRow("RF Trees:", self.spin_estimators)

        self.spin_depth = QSpinBox()
        self.spin_depth.setRange(0, 100)
        self.spin_depth.setValue(0)
        self.spin_depth.setToolTip("0 means no max depth")
        form.addRow("Max Depth:", self.spin_depth)

        self.spin_test = QDoubleSpinBox()
        self.spin_test.setRange(0.05, 0.5)
        self.spin_test.setSingleStep(0.05)
        self.spin_test.setValue(0.2)
        form.addRow("Test Split:", self.spin_test)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        form.addRow("Random Seed:", self.spin_seed)

        btn_row = QHBoxLayout()
        self.btn_train = QPushButton("Train Random Forest")
        self.btn_train.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 8px;")
        self.btn_train.clicked.connect(self.train_model)
        btn_row.addWidget(self.btn_train)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        btn_row.addWidget(self.lbl_status)
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("QTextEdit { font-family: Consolas; font-size: 12px; }")
        main_layout.addWidget(self.log)

    def set_status(self, text):
        self.lbl_status.setText(f"Status: {text}")
        self.log.append(text)
        QApplication.processEvents()

    def browse_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Raw CSV", self.input_csv.text(), "CSV Files (*.csv)")
        if p:
            self.input_csv.setText(p)

    def browse_model(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save RF Model", self.input_model.text(), "Joblib Files (*.joblib)"
        )
        if p:
            self.input_model.setText(p)

    @staticmethod
    def load_raw_segments(csv_path):
        required = {"Label", "Trial_ID", "Ch1", "Ch2", "Ch3", "Ch4"}
        segments = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")
            missing = sorted(list(required - set(reader.fieldnames)))
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            current_key = None
            current_label = None
            buf = []
            for row in reader:
                label = str(row["Label"]).strip()
                trial = str(row.get("Trial_ID", "")).strip()
                phase = str(row.get("Phase_ID", "")).strip()
                key = (label, trial, phase)
                sample = [
                    float(row["Ch1"]),
                    float(row["Ch2"]),
                    float(row["Ch3"]),
                    float(row["Ch4"]),
                ]

                if current_key is None:
                    current_key = key
                    current_label = label

                if key != current_key:
                    if len(buf) > 0:
                        segments.append((current_label, np.asarray(buf, dtype=np.float32)))
                    buf = []
                    current_key = key
                    current_label = label

                buf.append(sample)

            if len(buf) > 0:
                segments.append((current_label, np.asarray(buf, dtype=np.float32)))

        return segments

    def build_dataset(self, segments, sample_rate, win_samples, stride_samples):
        X = []
        y = []
        for label, seq in segments:
            windows = build_windows_from_sequence(seq, win_samples, stride_samples)
            for w in windows:
                X.append(extract_window_features(w, sample_rate))
                y.append(label)

        if len(X) == 0:
            raise ValueError("No training windows generated. Increase data length or reduce window size.")

        return np.asarray(X, dtype=np.float32), np.asarray(y)

    def train_model(self):
        if not HAS_SKLEARN:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "scikit-learn is required.\nInstall with:\npy -3 -m pip install scikit-learn",
            )
            return

        csv_path = self.input_csv.text().strip()
        model_path = self.input_model.text().strip()
        if not os.path.isfile(csv_path):
            QMessageBox.warning(self, "Input Error", f"Raw CSV not found:\n{csv_path}")
            return

        self.btn_train.setEnabled(False)
        start_t = time.time()
        try:
            sr = int(self.spin_sr.value())
            win_samples = int((self.spin_win.value() / 1000.0) * sr)
            stride_samples = int((self.spin_stride.value() / 1000.0) * sr)
            win_samples = max(8, win_samples)
            stride_samples = max(1, stride_samples)

            self.set_status("Loading raw CSV...")
            segments = self.load_raw_segments(csv_path)
            self.set_status(f"Loaded segments: {len(segments)}")

            self.set_status("Building windows + features...")
            X, y = self.build_dataset(segments, sr, win_samples, stride_samples)
            self.set_status(f"Dataset built: X={X.shape}, labels={len(np.unique(y))}")

            classes = sorted(list(np.unique(y)))
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_idx = np.asarray([class_to_idx[v] for v in y], dtype=np.int32)

            self.set_status("Splitting train/test...")
            test_size = float(self.spin_test.value())
            seed = int(self.spin_seed.value())
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=test_size, random_state=seed, stratify=y_idx
            )

            max_depth = int(self.spin_depth.value())
            max_depth = None if max_depth <= 0 else max_depth

            self.set_status("Training Random Forest...")
            model = RandomForestClassifier(
                n_estimators=int(self.spin_estimators.value()),
                max_depth=max_depth,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            self.set_status("Evaluating model...")
            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            report = classification_report(y_test, y_pred, target_names=classes, digits=4)
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))

            artifact = {
                "model": model,
                "class_names": classes,
                "sample_rate": sr,
                "window_samples": win_samples,
                "stride_samples": stride_samples,
                "feature_names": feature_names(),
                "feature_module": "rf_features.extract_window_features",
                "created_at_unix": time.time(),
                "created_at_text": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            joblib.dump(artifact, model_path)

            elapsed = time.time() - start_t
            summary = (
                f"Saved model: {model_path}\n"
                f"Accuracy: {acc:.4f}\n"
                f"Train size: {len(y_train)} | Test size: {len(y_test)}\n"
                f"Features: {X.shape[1]} | Classes: {', '.join(classes)}\n"
                f"Elapsed: {elapsed:.1f}s"
            )
            self.set_status("Training complete.")

            dlg = ResultsDialog(summary, report, cm, classes, self)
            dlg.exec_()

        except Exception as e:
            err = f"Training failed: {e}\n\n{traceback.format_exc()}"
            self.set_status("Training failed.")
            QMessageBox.critical(self, "Error", err)
        finally:
            self.btn_train.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = RFTrainerApp()
    w.show()
    sys.exit(app.exec_())
