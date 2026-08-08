import math
import random
import socket
import time

import serial
import serial.tools.list_ports
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from app_theme import apply_dark_theme, themed_button_style

DEFAULT_BAUD_RATE = 921600
DEFAULT_SAMPLE_RATE = 500
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 7000
MAX_CHANNELS = 9


class SimulatorWorker(QThread):
    stats_updated = pyqtSignal(str, float)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        output_mode,
        serial_port,
        baud_rate,
        num_channels,
        sample_rate,
        noise_std,
        tcp_host,
        tcp_port,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.serial_port = serial_port
        self.baud_rate = int(baud_rate)
        self.num_channels = int(num_channels)
        self.sample_rate = max(1, int(sample_rate))
        self.noise_std = max(0.1, float(noise_std))
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self._running = True

        self._phase_plan = self._create_phase_plan(self.num_channels)

    def stop(self):
        self._running = False
        self.wait()

    @staticmethod
    def _create_phase_plan(num_channels):
        plan = [("REST", 4.0, [0] * num_channels, 8.0)]
        for idx in range(num_channels):
            active = [0] * num_channels
            active[idx] = 1
            plan.append((f"FLEX_CH{idx + 1}", 3.0, active, 220.0))

        half = max(1, num_channels // 2)
        first_half = [1 if i < half else 0 for i in range(num_channels)]
        second_half = [1 if i >= half else 0 for i in range(num_channels)]
        all_active = [1] * num_channels
        plan.append(("GROUP_A", 3.0, first_half, 160.0))
        plan.append(("GROUP_B", 3.0, second_half, 160.0))
        plan.append(("WAVE_ALL", 4.0, all_active, 130.0))
        return plan

    def _phase_sample(self, t, active_flags, amp, base_levels):
        values = []
        emg1 = math.sin(2.0 * math.pi * 35.0 * t)
        emg2 = math.sin(2.0 * math.pi * 70.0 * t)
        for ch in range(self.num_channels):
            active = active_flags[ch] if ch < len(active_flags) else 0
            drift = 18.0 * math.sin(2.0 * math.pi * (0.08 + ch * 0.02) * t)
            noise = random.gauss(0.0, self.noise_std)
            burst = 0.0
            if active:
                envelope = 0.60 + 0.40 * math.sin(2.0 * math.pi * 0.6 * t + ch)
                burst = amp * envelope * (0.55 * emg1 + 0.35 * emg2 + 0.10 * random.random())
            val = base_levels[ch] + drift + noise + burst
            val = max(0.0, min(4095.0, val))
            values.append(val)
        return values

    def _build_line(self, now, t0, phase_idx, phase_start, base_levels):
        phase_name, phase_duration, active_flags, amp = self._phase_plan[phase_idx]
        elapsed_in_phase = now - phase_start
        if elapsed_in_phase >= phase_duration:
            phase_idx = (phase_idx + 1) % len(self._phase_plan)
            phase_name, phase_duration, active_flags, amp = self._phase_plan[phase_idx]
            phase_start = now

        t = now - t0
        sample = self._phase_sample(t, active_flags, amp, base_levels)
        line = ",".join(f"{v:.2f}" for v in sample[: self.num_channels]) + "\n"
        return line.encode("utf-8"), phase_name, phase_idx, phase_start

    def run(self):
        if self.output_mode == "tcp_server":
            self._run_tcp_server()
        else:
            self._run_serial()

    def _run_serial(self):
        try:
            ser = serial.Serial(
                self.serial_port,
                self.baud_rate,
                timeout=0.05,
                write_timeout=0.05,
            )
            self.status_updated.emit(f"Streaming on {self.serial_port}")
        except Exception as exc:
            self.error_occurred.emit(f"Failed to open serial port: {exc}")
            return

        base_levels = [1980.0 + i * 30.0 for i in range(self.num_channels)]
        rate_counter = 0
        rate_t0 = time.perf_counter()
        t0 = rate_t0
        phase_idx = 0
        phase_start = t0
        next_tick = t0

        try:
            while self._running:
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(min(0.001, next_tick - now))
                    continue

                payload, phase_name, phase_idx, phase_start = self._build_line(
                    now, t0, phase_idx, phase_start, base_levels
                )
                try:
                    ser.write(payload)
                except Exception as exc:
                    self.error_occurred.emit(f"Serial write failed: {exc}")
                    return

                rate_counter += 1
                elapsed_rate = now - rate_t0
                if elapsed_rate >= 1.0:
                    self.stats_updated.emit(phase_name, rate_counter / elapsed_rate)
                    rate_counter = 0
                    rate_t0 = now

                next_tick += 1.0 / self.sample_rate
                if now - next_tick > 0.5:
                    next_tick = now
        finally:
            try:
                ser.close()
            except Exception:
                pass

    def _run_tcp_server(self):
        server = None
        client = None
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.tcp_host, self.tcp_port))
            server.listen(1)
            server.settimeout(0.25)
            self.status_updated.emit(
                f"TCP listening on {self.tcp_host}:{self.tcp_port} (waiting for main.py)"
            )
        except Exception as exc:
            self.error_occurred.emit(f"Failed to start TCP server: {exc}")
            return

        base_levels = [1980.0 + i * 30.0 for i in range(self.num_channels)]
        rate_counter = 0
        rate_t0 = time.perf_counter()
        t0 = rate_t0
        phase_idx = 0
        phase_start = t0
        next_tick = t0

        try:
            while self._running:
                if client is None:
                    try:
                        client, addr = server.accept()
                        client.settimeout(0.5)
                        self.status_updated.emit(f"Client connected: {addr[0]}:{addr[1]}")
                    except socket.timeout:
                        continue
                    except Exception as exc:
                        self.error_occurred.emit(f"Accept failed: {exc}")
                        return

                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(min(0.001, next_tick - now))
                    continue

                payload, phase_name, phase_idx, phase_start = self._build_line(
                    now, t0, phase_idx, phase_start, base_levels
                )
                try:
                    client.sendall(payload)
                except Exception:
                    try:
                        client.close()
                    except Exception:
                        pass
                    client = None
                    self.status_updated.emit(
                        f"TCP listening on {self.tcp_host}:{self.tcp_port} (waiting for reconnect)"
                    )
                    continue

                rate_counter += 1
                elapsed_rate = now - rate_t0
                if elapsed_rate >= 1.0:
                    self.stats_updated.emit(phase_name, rate_counter / elapsed_rate)
                    rate_counter = 0
                    rate_t0 = now

                next_tick += 1.0 / self.sample_rate
                if now - next_tick > 0.5:
                    next_tick = now
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            if server is not None:
                try:
                    server.close()
                except Exception:
                    pass


class SimulatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Data Simulator")
        self.resize(700, 300)
        self.worker = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        form = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("TCP Server (Recommended)", "tcp_server")
        self.mode_combo.addItem("Serial COM", "serial")
        self.mode_combo.currentIndexChanged.connect(self._update_mode_controls)
        form.addRow("Output Mode", self.mode_combo)

        self.port_combo = QComboBox()
        self.refresh_ports()
        form.addRow("Serial Port", self.port_combo)

        self.tcp_host_edit = QLineEdit(DEFAULT_TCP_HOST)
        form.addRow("TCP Host", self.tcp_host_edit)

        self.tcp_port_spin = QSpinBox()
        self.tcp_port_spin.setRange(1024, 65535)
        self.tcp_port_spin.setValue(DEFAULT_TCP_PORT)
        form.addRow("TCP Port", self.tcp_port_spin)

        self.baud_spin = QSpinBox()
        self.baud_spin.setRange(1200, 4000000)
        self.baud_spin.setValue(DEFAULT_BAUD_RATE)
        form.addRow("Baud Rate", self.baud_spin)

        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, MAX_CHANNELS)
        self.channels_spin.setValue(4)
        form.addRow("Channels", self.channels_spin)

        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(20, 2000)
        self.sample_rate_spin.setValue(DEFAULT_SAMPLE_RATE)
        form.addRow("Sample Rate (Hz)", self.sample_rate_spin)

        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.1, 100.0)
        self.noise_spin.setDecimals(1)
        self.noise_spin.setSingleStep(0.5)
        self.noise_spin.setValue(6.0)
        form.addRow("Noise StdDev", self.noise_spin)

        layout.addLayout(form)

        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Ports")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        row.addWidget(self.btn_refresh)

        self.btn_toggle = QPushButton("Start Simulator")
        self.btn_toggle.setStyleSheet(themed_button_style("success"))
        self.btn_toggle.clicked.connect(self.toggle_stream)
        row.addWidget(self.btn_toggle)

        row.addStretch()
        layout.addLayout(row)

        self.lbl_status = QLabel("Status: Idle")
        self.lbl_phase = QLabel("Phase: -")
        self.lbl_rate = QLabel("Output Rate: 0.0 Hz")
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_phase)
        layout.addWidget(self.lbl_rate)

        tip = QLabel(
            "Use TCP mode: run simulator, then in main.py choose socket://127.0.0.1:7000 and connect."
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._tick_idle)
        self.ui_timer.start(500)

        self._update_mode_controls()

    def _tick_idle(self):
        if self.worker is None:
            self.lbl_phase.setText("Phase: -")
            self.lbl_rate.setText("Output Rate: 0.0 Hz")

    def _update_mode_controls(self):
        mode = self.mode_combo.currentData()
        is_serial = mode == "serial"
        self.port_combo.setEnabled(is_serial and self.worker is None)
        self.btn_refresh.setEnabled(is_serial and self.worker is None)
        self.baud_spin.setEnabled(is_serial and self.worker is None)
        self.tcp_host_edit.setEnabled((not is_serial) and self.worker is None)
        self.tcp_port_spin.setEnabled((not is_serial) and self.worker is None)

    def refresh_ports(self):
        current = self.port_combo.currentData()
        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            self.port_combo.addItem("No ports found", None)
            return
        for p in ports:
            self.port_combo.addItem(f"{p.device} - {p.description}", p.device)
        if current is not None:
            idx = self.port_combo.findData(current)
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)

    def _set_controls_enabled(self, enabled):
        self.mode_combo.setEnabled(enabled)
        self.channels_spin.setEnabled(enabled)
        self.sample_rate_spin.setEnabled(enabled)
        self.noise_spin.setEnabled(enabled)
        self._update_mode_controls()

    def toggle_stream(self):
        if self.worker is None:
            self.start_stream()
        else:
            self.stop_stream()

    def start_stream(self):
        mode = self.mode_combo.currentData()
        serial_port = self.port_combo.currentData()

        if mode == "serial" and not serial_port:
            QMessageBox.warning(self, "No Port", "Select a valid serial port or use TCP mode.")
            return

        self.worker = SimulatorWorker(
            output_mode=mode,
            serial_port=serial_port,
            baud_rate=self.baud_spin.value(),
            num_channels=self.channels_spin.value(),
            sample_rate=self.sample_rate_spin.value(),
            noise_std=self.noise_spin.value(),
            tcp_host=self.tcp_host_edit.text().strip() or DEFAULT_TCP_HOST,
            tcp_port=self.tcp_port_spin.value(),
        )
        self.worker.stats_updated.connect(self.on_stats)
        self.worker.status_updated.connect(self.on_status)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        self._set_controls_enabled(False)
        self.btn_toggle.setText("Stop Simulator")
        self.btn_toggle.setStyleSheet(themed_button_style("danger"))
        if mode == "tcp_server":
            self.lbl_status.setText(
                f"Status: Starting TCP server on {self.tcp_host_edit.text().strip() or DEFAULT_TCP_HOST}:{self.tcp_port_spin.value()}"
            )
        else:
            self.lbl_status.setText(f"Status: Streaming on {serial_port}")

    def stop_stream(self):
        if self.worker:
            self.worker.stop()

    def on_status(self, text):
        self.lbl_status.setText(f"Status: {text}")

    def on_stats(self, phase_name, rate_hz):
        self.lbl_phase.setText(f"Phase: {phase_name}")
        self.lbl_rate.setText(f"Output Rate: {rate_hz:.1f} Hz")

    def on_worker_error(self, msg):
        QMessageBox.critical(self, "Simulator Error", msg)
        self.lbl_status.setText(f"Status: Error - {msg}")
        self.stop_stream()

    def on_worker_finished(self):
        self.worker = None
        self._set_controls_enabled(True)
        self.btn_toggle.setText("Start Simulator")
        self.btn_toggle.setStyleSheet(themed_button_style("success"))
        if not self.lbl_status.text().startswith("Status: Error"):
            self.lbl_status.setText("Status: Idle")

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


def main():
    app = QApplication([])
    apply_dark_theme(app, font_size=16)
    window = SimulatorWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
