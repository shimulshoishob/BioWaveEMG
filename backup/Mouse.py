import sys
import os
import time
import threading
import numpy as np
import joblib
import serial
import serial.tools.list_ports

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox, QFileDialog, QFrame,
    QGroupBox, QSpinBox, QDoubleSpinBox, QScrollArea
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QColor

# Try to import feature extraction from BioWave
try:
    from rf_features import extract_window_features

    HAS_RF = True
except ImportError:
    HAS_RF = False

# Try to import Mouse control library
try:
    import pyautogui

    pyautogui.FAILSAFE = True  # Slam mouse to corner to abort
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

# Available Mouse Actions
MOUSE_ACTIONS = [
    "Ignore",
    "Move Up", "Move Down", "Move Left", "Move Right",
    "Left Click", "Right Click", "Double Click"
]


# --- BACKGROUND THREADS ---

class SerialWorker(QThread):
    """Reads live EMG data from the serial port or simulator."""
    batch_received = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, port_name, baud_rate, num_channels, batch_size=25):
        super().__init__()
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.num_channels = num_channels
        self.batch_size = batch_size
        self._running = True
        self._serial = None

    def run(self):
        partial_line = ""
        batch = []
        try:
            self._serial = serial.serial_for_url(self.port_name, self.baud_rate, timeout=0.05)
            self._serial.reset_input_buffer()

            while self._running:
                waiting = self._serial.in_waiting
                chunk = self._serial.read(waiting if waiting else 1)
                if not chunk:
                    continue

                partial_line += chunk.decode("utf-8", errors="ignore")
                lines = partial_line.split("\n")
                partial_line = lines.pop()

                for raw_line in lines:
                    line = raw_line.strip()
                    if not line:
                        continue

                    parts = line.replace(",", " ").split()
                    if len(parts) < self.num_channels:
                        continue

                    try:
                        vals = [float(parts[i]) for i in range(self.num_channels)]
                        batch.append(vals)
                    except ValueError:
                        continue

                    if len(batch) >= self.batch_size:
                        self.batch_received.emit(np.asarray(batch, dtype=np.float32))
                        batch = []
        except Exception as e:
            if self._running:
                self.error_occurred.emit(str(e))
        finally:
            if self._serial and self._serial.is_open:
                self._serial.close()

    def stop(self):
        self._running = False
        self.wait()


class InferenceWorker(QThread):
    """Extracts features and runs the Random Forest prediction."""
    prediction_ready = pyqtSignal(str, float)

    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.model = None
        self.class_names = []
        self._window = None
        self._running = True
        self._lock = threading.Lock()
        self._event = threading.Event()

    def load_model(self, model, class_names):
        with self._lock:
            self.model = model
            self.class_names = class_names

    def submit_window(self, window):
        with self._lock:
            self._window = window
        self._event.set()

    def run(self):
        while self._running:
            self._event.wait(0.1)
            if not self._running:
                break
            if not self._event.is_set():
                continue
            self._event.clear()

            with self._lock:
                win = self._window
                model = self.model
                classes = self.class_names
                self._window = None

            if win is None or model is None or not HAS_RF:
                continue

            try:
                feats = extract_window_features(win, sample_rate=self.sample_rate).reshape(1, -1)

                conf = 0.0
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feats)[0]
                    pred_idx = int(np.argmax(proba))
                    conf = float(proba[pred_idx])
                    model_classes = list(getattr(model, "classes_", []))
                    best_cls = model_classes[pred_idx]

                    if isinstance(best_cls, (int, np.integer)) and 0 <= best_cls < len(classes):
                        pred_label = classes[best_cls]
                    else:
                        pred_label = str(best_cls)
                else:
                    pred_raw = model.predict(feats)[0]
                    if isinstance(pred_raw, (int, np.integer)) and 0 <= pred_raw < len(classes):
                        pred_label = classes[pred_raw]
                    else:
                        pred_label = str(pred_raw)
                        conf = 1.0

                self.prediction_ready.emit(pred_label, conf)

            except Exception as e:
                print(f"Inference error: {e}")

    def stop(self):
        self._running = False
        self._event.set()
        self.wait()


# --- MAIN APPLICATION UI ---

class MouseControllerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioWave - Adaptive Mouse Controller")
        self.resize(550, 750)
        self.apply_dark_theme()

        if not HAS_RF:
            QMessageBox.critical(self, "Missing File", "rf_features.py must be in the same folder!")
            sys.exit(1)

        # State variables
        self.serial_worker = None
        self.inference_worker = None
        self.is_connected = False
        self.model_loaded = False
        self.mouse_control_active = False

        # Buffer Metadata
        self.num_channels = 4
        self.sample_rate = 500
        self.window_samples = 100
        self.stride_samples = 25
        self.samples_since_last_pred = 0
        self.data_buffer = None
        self.baseline_offsets = None

        # Mouse logic
        self.class_action_map = {}  # { "class_name" : "Action" }
        self.mapping_combos = []  # UI references
        self.last_click_time = 0.0

        self.init_ui()

        self.inference_worker = InferenceWorker(self.sample_rate)
        self.inference_worker.prediction_ready.connect(self.on_prediction_ready)
        self.inference_worker.start()

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #132440; color: #E8EEF0; font-family: 'Segoe UI', Arial; font-size: 14px; }
            QGroupBox { border: 1px solid #3B9797; border-radius: 6px; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; color: #3B9797; }
            QPushButton { background-color: #3B9797; color: white; border-radius: 4px; padding: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #4ebfbf; }
            QPushButton:disabled { background-color: #16476A; color: #6F8A99; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #16476A; border: 1px solid #3B9797; border-radius: 4px; padding: 4px; color: white; }
            QScrollArea { border: none; }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)

        # 1. Connection & Model Group
        grp_setup = QGroupBox("1. Setup")
        setup_layout = QVBoxLayout(grp_setup)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("RF Model:"))
        self.txt_model_path = QLineEdit()
        self.txt_model_path.setReadOnly(True)
        model_row.addWidget(self.txt_model_path)
        btn_browse = QPushButton("Browse .joblib")
        btn_browse.clicked.connect(self.browse_model)
        model_row.addWidget(btn_browse)
        setup_layout.addLayout(model_row)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Data Stream:"))
        self.combo_ports = QComboBox()
        self.refresh_ports()
        port_row.addWidget(self.combo_ports, 1)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_ports)
        port_row.addWidget(btn_refresh)
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.toggle_connection)
        port_row.addWidget(self.btn_connect)
        setup_layout.addLayout(port_row)
        layout.addWidget(grp_setup)

        # 2. Action Mapping Group
        grp_map = QGroupBox("2. Class to Action Mapping")
        map_layout = QVBoxLayout(grp_map)

        # Scroll area for dynamic classes
        self.scroll_map = QScrollArea()
        self.scroll_map.setWidgetResizable(True)
        self.map_content = QWidget()
        self.map_form = QFormLayout(self.map_content)
        self.scroll_map.setWidget(self.map_content)
        map_layout.addWidget(self.scroll_map)

        lbl_hint = QLabel("<i>Load a model to view your classes.</i>")
        lbl_hint.setStyleSheet("color: #A9C2CF;")
        self.map_form.addRow(lbl_hint)
        layout.addWidget(grp_map, 1)  # Give it stretch priority

        # 3. Settings Group
        grp_settings = QGroupBox("3. Control Settings")
        set_layout = QFormLayout(grp_settings)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(10.0, 99.9)
        self.spin_conf.setValue(65.0)
        self.spin_conf.setSuffix("%")
        set_layout.addRow("Minimum Confidence:", self.spin_conf)

        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 150)
        self.spin_speed.setValue(30)
        self.spin_speed.setSuffix(" px")
        set_layout.addRow("Mouse Speed (Per Tick):", self.spin_speed)

        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(0.1, 5.0)
        self.spin_cooldown.setValue(1.0)
        self.spin_cooldown.setSuffix(" sec")
        set_layout.addRow("Click Cooldown:", self.spin_cooldown)
        layout.addWidget(grp_settings)

        # 4. Status & Control
        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #A9C2CF;")
        layout.addWidget(self.lbl_status)

        self.lbl_prediction = QLabel("REST")
        self.lbl_prediction.setAlignment(Qt.AlignCenter)
        self.lbl_prediction.setFont(QFont("Arial", 28, QFont.Bold))
        self.lbl_prediction.setStyleSheet("color: #6F8A99;")
        layout.addWidget(self.lbl_prediction)

        self.lbl_conf = QLabel("Conf: 0.0%")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_conf)

        self.btn_mouse_toggle = QPushButton("ENABLE MOUSE CONTROL")
        self.btn_mouse_toggle.setStyleSheet("background-color: #2e7d32; font-size: 16px; padding: 12px;")
        self.btn_mouse_toggle.setCheckable(True)
        self.btn_mouse_toggle.setEnabled(False)
        self.btn_mouse_toggle.toggled.connect(self.toggle_mouse_control)
        layout.addWidget(self.btn_mouse_toggle)

        lbl_safety = QLabel("<b>Safety Feature:</b> Move physical mouse to screen corner to abort!")
        lbl_safety.setAlignment(Qt.AlignCenter)
        lbl_safety.setStyleSheet("color: #BF092F; font-size: 11px;")
        layout.addWidget(lbl_safety)

        if not HAS_PYAUTOGUI:
            QMessageBox.warning(self, "Missing Library",
                                "pyautogui not found. Please 'pip install pyautogui' for mouse control.")

    def refresh_ports(self):
        self.combo_ports.clear()
        self.combo_ports.addItem("socket://127.0.0.1:7000 (Simulator)", "socket://127.0.0.1:7000")
        for p in serial.tools.list_ports.comports():
            self.combo_ports.addItem(f"{p.device} - {p.description}", p.device)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select RF Model", "", "Joblib Files (*.joblib)")
        if path:
            try:
                artifact = joblib.load(path)
                model = artifact["model"]
                class_names = artifact["class_names"]
                self.window_samples = artifact.get("window_samples", 100)
                self.stride_samples = artifact.get("stride_samples", 25)
                self.num_channels = artifact.get("input_channels", 4)

                self.inference_worker.load_model(model, class_names)
                self.txt_model_path.setText(path)
                self.build_mapping_ui(class_names)

                self.model_loaded = True
                self.check_ready_state()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{e}")

    def build_mapping_ui(self, class_names):
        # Clear existing layout
        while self.map_form.count():
            item = self.map_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.mapping_combos = []
        self.class_action_map = {}

        for cls in class_names:
            combo = QComboBox()
            combo.addItems(MOUSE_ACTIONS)

            # Auto-guess mapping based on naming conventions
            clower = cls.lower()
            if "up" in clower:
                combo.setCurrentText("Move Up")
            elif "down" in clower:
                combo.setCurrentText("Move Down")
            elif "left" in clower and "click" not in clower:
                combo.setCurrentText("Move Left")
            elif "right" in clower and "click" not in clower:
                combo.setCurrentText("Move Right")
            elif "double" in clower:
                combo.setCurrentText("Double Click")
            elif "right" in clower and "click" in clower:
                combo.setCurrentText("Right Click")
            elif "click" in clower or "fist" in clower:
                combo.setCurrentText("Left Click")
            else:
                combo.setCurrentText("Ignore")

            # Connect change event
            combo.currentTextChanged.connect(lambda text, c=cls: self.update_map(c, text))
            self.update_map(cls, combo.currentText())  # Initial population

            self.map_form.addRow(f"Gesture: <b>{cls}</b>", combo)
            self.mapping_combos.append(combo)

    def update_map(self, cls, action):
        self.class_action_map[cls] = action

    def toggle_connection(self):
        if self.is_connected:
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self):
        port = self.combo_ports.currentData()
        if not port:
            port = self.combo_ports.currentText().split()[0]

        if not self.model_loaded:
            QMessageBox.warning(self, "Model Required", "Please load a model first.")
            return

        self.data_buffer = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)
        self.baseline_offsets = np.zeros(self.num_channels, dtype=np.float32)
        self.samples_since_last_pred = 0

        self.serial_worker = SerialWorker(port, 921600, self.num_channels, batch_size=self.stride_samples)
        self.serial_worker.batch_received.connect(self.on_batch_received)
        self.serial_worker.error_occurred.connect(lambda e: QMessageBox.warning(self, "Serial Error", e))
        self.serial_worker.start()

        self.is_connected = True
        self.btn_connect.setText("Disconnect")
        self.btn_connect.setStyleSheet("background-color: #BF092F;")
        self.check_ready_state()
        self.lbl_status.setText("Status: Streaming Live Data")
        self.lbl_status.setStyleSheet("color: #3B9797;")

    def disconnect_serial(self):
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker = None
        self.is_connected = False
        self.btn_connect.setText("Connect")
        self.btn_connect.setStyleSheet("")  # Revert to default
        self.btn_mouse_toggle.setChecked(False)
        self.check_ready_state()
        self.lbl_status.setText("Status: Disconnected")
        self.lbl_status.setStyleSheet("color: #A9C2CF;")

    def check_ready_state(self):
        ready = self.model_loaded and self.is_connected
        self.btn_mouse_toggle.setEnabled(ready and HAS_PYAUTOGUI)
        if not ready and self.btn_mouse_toggle.isChecked():
            self.btn_mouse_toggle.setChecked(False)

    def toggle_mouse_control(self, checked):
        self.mouse_control_active = checked and HAS_PYAUTOGUI
        if checked:
            self.btn_mouse_toggle.setText("STOP MOUSE CONTROL")
            self.btn_mouse_toggle.setStyleSheet("background-color: #BF092F; font-size: 16px; padding: 12px;")
        else:
            self.btn_mouse_toggle.setText("ENABLE MOUSE CONTROL")
            self.btn_mouse_toggle.setStyleSheet("background-color: #2e7d32; font-size: 16px; padding: 12px;")

    def on_batch_received(self, batch):
        if self.data_buffer is None: return

        # Adaptive baseline centering
        for ch in range(self.num_channels):
            mean_val = np.mean(batch[:, ch])
            self.baseline_offsets[ch] = 0.95 * self.baseline_offsets[ch] + 0.05 * mean_val

        centered_batch = batch - self.baseline_offsets[np.newaxis, :]
        new_data = centered_batch.T
        num_new = new_data.shape[1]

        self.data_buffer[:, :-num_new] = self.data_buffer[:, num_new:]
        self.data_buffer[:, -num_new:] = new_data

        self.samples_since_last_pred += num_new
        if self.samples_since_last_pred >= self.stride_samples:
            self.samples_since_last_pred = 0
            self.inference_worker.submit_window(self.data_buffer.copy())

    def on_prediction_ready(self, label, conf):
        conf_pct = conf * 100.0
        req_conf = self.spin_conf.value()

        # Update UI text
        self.lbl_prediction.setText(label.upper())
        self.lbl_conf.setText(f"Conf: {conf_pct:.1f}%")

        if conf_pct < req_conf or self.class_action_map.get(label) == "Ignore":
            self.lbl_prediction.setStyleSheet("color: #6F8A99;")
            return  # Below confidence threshold or mapped to Ignore

        self.lbl_prediction.setStyleSheet("color: #3B9797;")  # Active color

        # Process Mouse Action
        if self.mouse_control_active:
            action = self.class_action_map.get(label, "Ignore")
            self.execute_mouse_action(action)

    def execute_mouse_action(self, action):
        if action == "Ignore":
            return

        speed = self.spin_speed.value()
        now = time.time()

        try:
            if action == "Move Up":
                pyautogui.move(0, -speed)
            elif action == "Move Down":
                pyautogui.move(0, speed)
            elif action == "Move Left":
                pyautogui.move(-speed, 0)
            elif action == "Move Right":
                pyautogui.move(speed, 0)
            elif "Click" in action:
                # Check cooldown to prevent rapid-fire clicking
                if now - self.last_click_time > self.spin_cooldown.value():
                    if action == "Left Click":
                        pyautogui.click(button='left')
                    elif action == "Right Click":
                        pyautogui.click(button='right')
                    elif action == "Double Click":
                        pyautogui.doubleClick()

                    self.last_click_time = now
        except pyautogui.FailSafeException:
            # User slammed mouse into corner
            self.btn_mouse_toggle.setChecked(False)
            QMessageBox.critical(self, "Safety Triggered", "Mouse hit the screen corner. Control disabled for safety.")

    def closeEvent(self, event):
        self.disconnect_serial()
        if self.inference_worker:
            self.inference_worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MouseControllerApp()
    window.show()
    sys.exit(app.exec_())
