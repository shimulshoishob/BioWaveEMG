import sys
import os
import time
import socket
import struct
import hashlib
import hmac
import threading
from collections import deque
from dataclasses import dataclass
from urllib.parse import quote
from pathlib import Path
import numpy as np
import joblib
import serial
import serial.tools.list_ports

# Add known sibling paths to sys.path so rf_features can be imported when the file is elsewhere in the repo.
script_dir = Path(__file__).resolve().parent
candidate_dirs = [
    script_dir,
    script_dir.parent / "BioWave-GUI" / "code",
    script_dir.parent / "BioWave-GUI",
]
for candidate in candidate_dirs:
    if (candidate / "rf_features.py").exists():
        sys.path.insert(0, str(candidate))
        break

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox, QFileDialog, QFrame,
    QGroupBox, QSpinBox, QDoubleSpinBox, QScrollArea, QDialog, QTabWidget, QRadioButton,
    QButtonGroup, QStackedWidget, QSlider
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QIcon
from app_theme import apply_dark_theme as apply_app_theme, themed_button_style, themed_label_style

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
    pyautogui.PAUSE = 0.0      # Zero delay for continuous, smooth movement
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

# PyAutoGUI is convenient, but on macOS every call includes a small Darwin
# catch-up delay.  That delay is noticeable when a cursor event is emitted at
# 60+ Hz.  Quartz is already supplied by PyObjC on the standard Python.org
# macOS install, so prefer it for the real-time path and retain PyAutoGUI as a
# cross-platform fallback.
try:
    if sys.platform == "darwin":
        import Quartz
        HAS_QUARTZ_MOUSE = True
    else:
        Quartz = None
        HAS_QUARTZ_MOUSE = False
except ImportError:
    Quartz = None
    HAS_QUARTZ_MOUSE = False

HAS_MOUSE_CONTROL = HAS_QUARTZ_MOUSE or HAS_PYAUTOGUI


class MouseSafetyTriggered(RuntimeError):
    """Raised when the cursor reaches a display corner while control is active."""


class MouseBackend:
    """Low-latency mouse events, using Quartz on macOS where available."""

    def __init__(self):
        self.uses_quartz = HAS_QUARTZ_MOUSE

    def _quartz_position(self):
        event = Quartz.CGEventCreate(None)
        return Quartz.CGEventGetLocation(event)

    def _at_safety_corner(self, point):
        # Check every display so the documented corner abort works with
        # multiple monitors as well as the primary display.
        for display_id in Quartz.CGGetActiveDisplayList(32, None, None)[1]:
            bounds = Quartz.CGDisplayBounds(display_id)
            left, top = bounds.origin.x, bounds.origin.y
            right = left + bounds.size.width - 1
            bottom = top + bounds.size.height - 1
            if ((abs(point.x - left) <= 1 or abs(point.x - right) <= 1) and
                    (abs(point.y - top) <= 1 or abs(point.y - bottom) <= 1)):
                return True
        return False

    def move_relative(self, dx, dy):
        if self.uses_quartz:
            point = self._quartz_position()
            if self._at_safety_corner(point):
                raise MouseSafetyTriggered()
            target = (point.x + dx, point.y + dy)
            event = Quartz.CGEventCreateMouseEvent(
                None, Quartz.kCGEventMouseMoved, target, Quartz.kCGMouseButtonLeft
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            return
        pyautogui.moveRel(dx, dy, duration=0)

    def click(self, button="left", clicks=1):
        if self.uses_quartz:
            point = self._quartz_position()
            if self._at_safety_corner(point):
                raise MouseSafetyTriggered()
            button_map = {
                "left": (Quartz.kCGMouseButtonLeft, Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp),
                "right": (Quartz.kCGMouseButtonRight, Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp),
            }
            mouse_button, down_type, up_type = button_map[button]
            for click_number in range(1, clicks + 1):
                for event_type in (down_type, up_type):
                    event = Quartz.CGEventCreateMouseEvent(None, event_type, point, mouse_button)
                    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, click_number)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            return
        if clicks == 2:
            pyautogui.doubleClick(button=button, interval=0)
        else:
            pyautogui.click(button=button)

# Available Mouse Actions
MOUSE_ACTIONS = [
    "Ignore", 
    "Move Up", "Move Down", "Move Left", "Move Right", 
    "Left Click", "Right Click", "Double Click"
]

# Wireless & Communication Constants
WIFI_STREAM_PORT = 5000
WIFI_CONTROL_PORT = 5001
DEFAULT_ACCESS_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_KEY"
DISCOVERY_ADDRESS = "255.255.255.255"
DISCOVERY_TIMEOUT = 1.2
CONTROL_TIMEOUT = 2.0
USB_SERIAL_BAUD = 115200
SERIAL_BOOT_WAIT_S = 3.5
SERIAL_RESPONSE_TIMEOUT_S = 15.0

WIRELESS_EMG_CHANNELS = 8
WIRELESS_IMU_CHANNELS = 3
WIRELESS_TOTAL_CHANNELS = WIRELESS_EMG_CHANNELS + WIRELESS_IMU_CHANNELS
WIFI_PACKET_HEADER_FORMAT = "<4sBBHI"
WIFI_PACKET_HEADER_SIZE = struct.calcsize(WIFI_PACKET_HEADER_FORMAT)
WIRELESS_FRAME_FORMAT = "<IIII8HfffB3x"
WIRELESS_FRAME_SIZE = struct.calcsize(WIRELESS_FRAME_FORMAT)
WIRELESS_FRAMES_PER_PACKET = 5
WIRELESS_PACKET_SIZE = WIFI_PACKET_HEADER_SIZE + (WIRELESS_FRAME_SIZE * WIRELESS_FRAMES_PER_PACKET)

# --- NETWORK DATA STRUCTURES & PROTOCOLS ---

@dataclass
class DeviceInfo:
    ip: str
    device_id: str
    device_name: str
    wifi_mode: str
    reported_ip: str
    imu_ready: bool
    streaming: bool
    firmware: str

    @property
    def summary(self):
        return f"{self.device_name} @ {self.ip} ({self.wifi_mode})"


@dataclass
class SerialDeviceInfo:
    port_name: str
    device_id: str
    device_name: str
    imu_ready: bool
    wifi_saved: bool
    firmware: str


def sign_message(secret, *parts):
    message = "|".join(str(part) for part in parts)
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def get_broadcast_addresses():
    addresses = ["255.255.255.255"]
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = item[4][0]
            if ip and not ip.startswith("127."):
                parts = ip.split(".")
                if len(parts) == 4:
                    subnet_bc = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                    if subnet_bc not in addresses:
                        addresses.append(subnet_bc)
    except Exception:
        pass
    for prefix in ["192.168.1", "192.168.0", "192.168.4", "10.0.0"]:
        bc = f"{prefix}.255"
        if bc not in addresses:
            addresses.append(bc)
    return addresses


def get_local_ip_for_target(target_ip):
    # 1. Try UDP probe to target_ip directly
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect((target_ip, 1))
        ip = probe.getsockname()[0]
        probe.close()
        if ip and ip != "0.0.0.0" and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # 2. Try UDP probe to common DNS/gateway
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        probe.close()
        if ip and ip != "0.0.0.0" and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # 3. Hostname resolution
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = item[4][0]
            if ip and ip != "0.0.0.0" and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return "127.0.0.1"


class ControlProtocol:
    @staticmethod
    def parse_device_info(message, source_ip):
        parts = message.strip().split("|")
        if len(parts) != 8 or parts[0] != "HELLO":
            raise ValueError("Unexpected device response.")

        return DeviceInfo(
            ip=source_ip,
            device_id=parts[1],
            device_name=parts[2],
            wifi_mode=parts[3],
            reported_ip=parts[4],
            imu_ready=parts[5] == "1",
            streaming=parts[6] == "1",
            firmware=parts[7],
        )

    @staticmethod
    def send_and_receive(message, target_ip, expect_multiple=False, timeout=CONTROL_TIMEOUT, broadcast=False):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2 if expect_multiple else timeout)
        if broadcast:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        responses = []
        deadline = time.monotonic() + timeout
        try:
            if broadcast and target_ip == "255.255.255.255":
                for bc_ip in get_broadcast_addresses():
                    try:
                        sock.sendto(message.encode("utf-8"), (bc_ip, WIFI_CONTROL_PORT))
                    except OSError:
                        pass
            else:
                try:
                    sock.sendto(message.encode("utf-8"), (target_ip, WIFI_CONTROL_PORT))
                except OSError as exc:
                    if exc.errno == 65 or getattr(exc, "errno", None) == 65:
                        raise RuntimeError(f"No route to host ({target_ip}). Check network connection / Wi-Fi subnet.")
                    raise

            if expect_multiple:
                while time.monotonic() < deadline:
                    try:
                        data, addr = sock.recvfrom(2048)
                        responses.append((data.decode("utf-8", errors="replace"), addr[0]))
                    except socket.timeout:
                        continue
                    except OSError:
                        pass
                return responses

            data, addr = sock.recvfrom(2048)
            return data.decode("utf-8", errors="replace"), addr[0]
        except socket.timeout:
            raise RuntimeError(
                f"Connection timed out ({target_ip}).\n\n"
                "1. Ensure the ESP32 is powered on and connected to Wi-Fi.\n"
                "2. Verify the Access Key matches the device access key.\n"
                "3. Ensure PC and ESP32 are on the same Wi-Fi subnet (disable VPN if active)."
            )
        finally:
            sock.close()

    @staticmethod
    def discover():
        devices = {}
        responses = ControlProtocol.send_and_receive(
            "DISCOVER",
            DISCOVERY_ADDRESS,
            expect_multiple=True,
            timeout=DISCOVERY_TIMEOUT,
            broadcast=True,
        )
        for response, source_ip in responses:
            try:
                device = ControlProtocol.parse_device_info(response, source_ip)
                devices[device.ip] = device
            except ValueError:
                continue
        return list(devices.values())

    @staticmethod
    def get_challenge(target_ip):
        response, _ = ControlProtocol.send_and_receive("CHALLENGE", target_ip)
        parts = response.strip().split("|")
        if len(parts) != 2 or parts[0] != "CHALLENGE":
            raise RuntimeError("Device did not return a valid challenge.")
        return parts[1]

    @staticmethod
    def authenticated_command(target_ip, secret, command, *payload):
        if not secret:
            raise RuntimeError("Device access key is required.")

        challenge = ControlProtocol.get_challenge(target_ip)
        auth = sign_message(secret, command, challenge, *payload)
        message = "|".join([command, challenge, *payload, auth])
        response, _ = ControlProtocol.send_and_receive(message, target_ip)

        parts = response.strip().split("|")
        if not parts:
            raise RuntimeError("Device returned an empty response.")
        if parts[0] == "ERR":
            detail = parts[1] if len(parts) > 1 else "UNKNOWN"
            raise RuntimeError(f"Device rejected command: {detail}")
        if parts[0] != "ACK":
            raise RuntimeError("Unexpected device acknowledgement.")
        return parts[1:]

    @staticmethod
    def start_stream(target_ip, secret, client_ip, client_port):
        return ControlProtocol.authenticated_command(
            target_ip,
            secret,
            "START",
            client_ip,
            str(client_port),
        )

    @staticmethod
    def stop_stream(target_ip, secret):
        return ControlProtocol.authenticated_command(target_ip, secret, "STOP")

    @staticmethod
    def ping(target_ip, secret):
        return ControlProtocol.authenticated_command(target_ip, secret, "PING")


class WiFiSerialProvisionProtocol:
    @staticmethod
    def available_ports():
        return list(serial.tools.list_ports.comports())

    @staticmethod
    def _exchange_line(port_name, command, expected_prefixes, timeout=SERIAL_RESPONSE_TIMEOUT_S):
        try:
            with serial.Serial(port_name, USB_SERIAL_BAUD, timeout=0.3, write_timeout=1) as ser:
                ser.setDTR(False)
                ser.setRTS(False)
                time.sleep(0.15)
                time.sleep(SERIAL_BOOT_WAIT_S)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                ser.write((command + "\n").encode("utf-8"))
                ser.flush()

                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    raw_line = ser.readline()
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if any(line.startswith(prefix) for prefix in expected_prefixes):
                        return line
        except serial.SerialException as exc:
            raise RuntimeError(f"Serial communication failed on {port_name}: {exc}") from exc

        raise RuntimeError("The ESP32 did not return a serial response in time.")

    @staticmethod
    def query_info(port_name):
        response = WiFiSerialProvisionProtocol._exchange_line(port_name, "INFO", expected_prefixes=("INFO|", "ERR|"))
        parts = response.split("|")
        if len(parts) >= 2 and parts[0] == "ERR":
            raise RuntimeError(f"ESP32 returned an error: {parts[1]}")
        if len(parts) != 6 or parts[0] != "INFO":
            raise RuntimeError(f"Unexpected serial response: {response}")

        return SerialDeviceInfo(
            port_name=port_name,
            device_id=parts[1],
            device_name=parts[2],
            imu_ready=parts[3] == "1",
            wifi_saved=parts[4] == "1",
            firmware=parts[5],
        )

    @staticmethod
    def provision(port_name, ssid, password):
        encoded_ssid = quote(ssid, safe="")
        encoded_password = quote(password, safe="")
        response = WiFiSerialProvisionProtocol._exchange_line(
            port_name,
            f"PROVISION|{encoded_ssid}|{encoded_password}",
            expected_prefixes=("ACK|", "ERR|"),
            timeout=8.0,
        )
        parts = response.split("|")
        if len(parts) >= 2 and parts[0] == "ACK" and parts[1] == "PROVISIONED":
            return
        if len(parts) >= 2 and parts[0] == "ERR":
            raise RuntimeError(f"ESP32 rejected provisioning: {parts[1]}")
        raise RuntimeError(f"Unexpected serial response: {response}")


class USBProvisionDialog(QDialog):
    """Dialog for provisioning Wi-Fi credentials to ESP32 over USB serial."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("USB Wi-Fi Provisioning")
        self.setModal(True)
        self.resize(500, 320)
        self.selected_port = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QLabel("Connect ESP32-S3 over USB, select the serial port, probe, and send Wi-Fi credentials.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row_port = QHBoxLayout()
        row_port.addWidget(QLabel("Serial Port:"))
        self.port_selector = QComboBox()
        row_port.addWidget(self.port_selector, stretch=1)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_ports)
        row_port.addWidget(btn_refresh)
        layout.addLayout(row_port)

        form = QFormLayout()
        self.ssid_input = QLineEdit()
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.Password)
        form.addRow("Wi-Fi SSID:", self.ssid_input)
        form.addRow("Wi-Fi Password:", self.pass_input)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_probe = QPushButton("Probe Selected Port")
        btn_probe.clicked.connect(self.probe_selected_port)
        btn_row.addWidget(btn_probe)

        btn_send = QPushButton("Send Credentials")
        btn_send.setStyleSheet("background-color: #2e7d32; color: white;")
        btn_send.clicked.connect(self.send_credentials)
        btn_row.addWidget(btn_send)

        btn_cancel = QPushButton("Close")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color: #A9C2CF;")
        layout.addWidget(self.lbl_status)

        self.refresh_ports()

    def refresh_ports(self):
        self.port_selector.clear()
        ports = WiFiSerialProvisionProtocol.available_ports()
        for p in ports:
            self.port_selector.addItem(f"{p.device} - {p.description}", p.device)

    def probe_selected_port(self):
        port_name = self.port_selector.currentData() or self.port_selector.currentText().split()[0]
        if not port_name:
            QMessageBox.warning(self, "Warning", "Select a serial port first.")
            return
        try:
            info = WiFiSerialProvisionProtocol.query_info(port_name)
            self.lbl_status.setText(f"Detected {info.device_name} (ID: {info.device_id}, FW: {info.firmware})")
            self.lbl_status.setStyleSheet("color: #3B9797;")
        except Exception as e:
            QMessageBox.warning(self, "Probe Failed", str(e))
            self.lbl_status.setText("Probe failed.")

    def send_credentials(self):
        port_name = self.port_selector.currentData() or self.port_selector.currentText().split()[0]
        ssid = self.ssid_input.text().strip()
        password = self.pass_input.text()

        if not port_name or not ssid:
            QMessageBox.warning(self, "Warning", "Port and SSID are required.")
            return

        try:
            WiFiSerialProvisionProtocol.provision(port_name, ssid, password)
            QMessageBox.information(self, "Success", "Wi-Fi credentials sent to ESP32! It will now restart and join Wi-Fi.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Provision Error", str(e))


# --- BACKGROUND WORKERS ---

class KeepaliveWorker(QThread):
    """Sends background PING packets to ESP32 to prevent stream keepalive timeout."""
    ping_failed = pyqtSignal(str)

    def __init__(self, target_ip, access_key):
        super().__init__()
        self.target_ip = target_ip
        self.access_key = access_key
        self._running = True

    def run(self):
        while self._running:
            # Ping every 1.5s (ESP32 timeout is 5.0s)
            time.sleep(1.5)
            if not self._running:
                break
            try:
                ControlProtocol.ping(self.target_ip, self.access_key)
            except Exception as e:
                if self._running:
                    self.ping_failed.emit(str(e))

    def stop(self):
        self._running = False
        self.wait()


class WirelessStreamWorker(QThread):
    """Receives live UDP EMG+IMU datagrams from ESP32 hardware."""
    batch_received = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, port=WIFI_STREAM_PORT):
        super().__init__()
        self.port = int(port)
        self._running = True
        self._sock = None

    def _parse_datagram(self, data):
        # 1. Standard 252-byte binary packet with BWIM header
        if len(data) == WIRELESS_PACKET_SIZE:
            magic, version, frame_count, frame_size, packet_sequence = struct.unpack(
                WIFI_PACKET_HEADER_FORMAT, data[:WIFI_PACKET_HEADER_SIZE]
            )
            if magic != b"BWIM" or version != 1 or frame_size != WIRELESS_FRAME_SIZE:
                return None
            payload = data[WIFI_PACKET_HEADER_SIZE:]
            expected_payload = frame_count * frame_size
            if len(payload) != expected_payload:
                return None

            rows = []
            for offset in range(0, expected_payload, WIRELESS_FRAME_SIZE):
                frame = payload[offset : offset + WIRELESS_FRAME_SIZE]
                _frame_id, _frame_ts, _imu_id, _imu_ts, *frame_fields = struct.unpack(WIRELESS_FRAME_FORMAT, frame)
                # 8 EMG channels + roll, pitch, yaw
                row = [float(v) for v in frame_fields[:WIRELESS_EMG_CHANNELS]]
                row.extend([float(frame_fields[8]), float(frame_fields[9]), float(frame_fields[10])])
                rows.append(row)
            batch = np.asarray(rows, dtype=np.float32)
            return {"batch": batch, "source": "wireless"}

        # 2. Raw 240-byte multi-frame binary payload (without header)
        if len(data) == (WIRELESS_FRAME_SIZE * WIRELESS_FRAMES_PER_PACKET):
            rows = []
            for offset in range(0, len(data), WIRELESS_FRAME_SIZE):
                frame = data[offset : offset + WIRELESS_FRAME_SIZE]
                _frame_id, _frame_ts, _imu_id, _imu_ts, *frame_fields = struct.unpack(WIRELESS_FRAME_FORMAT, frame)
                row = [float(v) for v in frame_fields[:WIRELESS_EMG_CHANNELS]]
                row.extend([float(frame_fields[8]), float(frame_fields[9]), float(frame_fields[10])])
                rows.append(row)
            batch = np.asarray(rows, dtype=np.float32)
            return {"batch": batch, "source": "wireless_legacy"}

        # 3. Single 48-byte binary frame
        if len(data) == WIRELESS_FRAME_SIZE:
            _frame_id, _frame_ts, _imu_id, _imu_ts, *frame_fields = struct.unpack(WIRELESS_FRAME_FORMAT, data)
            row = [float(v) for v in frame_fields[:WIRELESS_EMG_CHANNELS]]
            row.extend([float(frame_fields[8]), float(frame_fields[9]), float(frame_fields[10])])
            batch = np.asarray([row], dtype=np.float32)
            return {"batch": batch, "source": "wireless_single"}

        # 4. Text CSV over UDP fallback
        try:
            line = data.decode("utf-8", errors="ignore").strip()
            if line:
                parts = line.replace(",", " ").split()
                if len(parts) >= 1:
                    vals = [float(p) for p in parts]
                    return {"batch": np.asarray([vals], dtype=np.float32), "source": "wireless_text"}
        except Exception:
            pass

        return None

    def run(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(("0.0.0.0", self.port))
        except Exception as e:
            self.error_occurred.emit(f"Failed to bind UDP stream port {self.port}: {e}")
            return
        self._sock.settimeout(1.0)

        try:
            while self._running:
                try:
                    data, _addr = self._sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError:
                    break

                payload = self._parse_datagram(data)
                if payload is not None:
                    batch = payload.get("batch")
                    if batch is not None and batch.size > 0:
                        self.batch_received.emit(payload)
        except Exception as exc:
            if self._running:
                self.error_occurred.emit(f"Wireless stream error: {exc}")
        finally:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self.wait()


class SerialWorker(QThread):
    """Reads live EMG data from serial port or TCP simulator."""
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
    """Extracts features and runs Random Forest prediction."""
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
        self.resize(580, 820)
        self.setMinimumSize(520, 640)
        
        if not HAS_RF:
            QMessageBox.critical(self, "Missing File", "rf_features.py must be in the same folder!")
            sys.exit(1)
            
        # State variables
        self.active_worker = None
        self.keepalive_worker = None
        self.inference_worker = None
        self.is_connected = False
        self.model_loaded = False
        self.mouse_control_active = False
        
        # Wireless state
        self.discovered_devices = []
        self.current_wireless_device = None
        self.wireless_access_key = DEFAULT_ACCESS_KEY

        # Buffer Metadata
        self.num_channels = 4
        self.sample_rate = 500
        self.window_samples = 100
        self.stride_samples = 25
        self.samples_since_last_pred = 0
        self.data_buffer = None
        self.baseline_offsets = None
        
        # Mouse logic, continuous motion & velocity smoothing
        self.class_action_map = {}  # { "class_name" : "Action" }
        self.mapping_combos = []
        self.last_click_time = 0.0
        self.last_clicked_action = ""
        self.pred_history = deque(maxlen=5) # Sliding window for prediction majority vote
        self.current_active_action = "Ignore"
        self.current_active_conf = 0.0

        # Sub-pixel accumulator & continuous analog velocity states
        self.curr_vx = 0.0
        self.curr_vy = 0.0
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.smooth_alpha = 0.35  # Acceleration / deceleration smoothing coefficient

        # A precise 120 Hz timer keeps native macOS cursor events responsive.
        # Velocity is expressed in pixels/second, so the perceived speed is
        # stable even when the event loop has a brief scheduling delay.
        self.smooth_motion_timer = QTimer(self)
        self.smooth_motion_timer.setTimerType(Qt.PreciseTimer)
        self.smooth_motion_timer.setInterval(8)
        self.smooth_motion_timer.timeout.connect(self.update_smooth_mouse_motion)
        self._last_motion_tick = time.monotonic()
        self.mouse_backend = MouseBackend()

        self.init_ui()
        
        self.inference_worker = InferenceWorker(self.sample_rate)
        self.inference_worker.prediction_ready.connect(self.on_prediction_ready)
        self.inference_worker.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("mouseHeader")
        header.setStyleSheet(
            "QFrame#mouseHeader { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #16102B, stop:0.55 #112B4A, stop:1 #0B192B); "
            "border: 1px solid #7C3AED; border-radius: 14px; }"
            "QFrame#mouseHeader QLabel { background: transparent; border: none; }"
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        title = QLabel("BIOWAVE  /  MOUSE CONTROL")
        title.setStyleSheet("font-size: 18px; font-weight: 800; letter-spacing: 1px; color: #F8FAFC;")
        subtitle = QLabel("EMG-DRIVEN CURSOR INTERFACE")
        subtitle.setStyleSheet("font-size: 10px; font-weight: 700; letter-spacing: 1px; color: #67E8F9;")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header)

        # 1. Setup Group (Model + Connection Medium Selector)
        grp_setup = QGroupBox("1. Connection & Model Setup")
        setup_layout = QVBoxLayout(grp_setup)
        
        # Model Selection Row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("RF Model:"))
        self.txt_model_path = QLineEdit()
        self.txt_model_path.setReadOnly(True)
        model_row.addWidget(self.txt_model_path)
        btn_browse = QPushButton("Browse .joblib")
        btn_browse.clicked.connect(self.browse_model)
        model_row.addWidget(btn_browse)
        setup_layout.addLayout(model_row)

        # Connection Tabs: Wireless vs Wired vs Simulator
        self.tab_medium = QTabWidget()
        
        # Tab 1: Wireless Wi-Fi (ESP32-S3)
        tab_wifi = QWidget()
        layout_wifi = QVBoxLayout(tab_wifi)
        
        row_dev = QHBoxLayout()
        row_dev.addWidget(QLabel("Hardware Device:"))
        self.combo_devices = QComboBox()
        self.combo_devices.setEditable(True)
        self.combo_devices.setPlaceholderText("Select or enter IP (e.g. 192.168.1.100)")
        row_dev.addWidget(self.combo_devices, stretch=1)
        btn_disc = QPushButton("Discover")
        btn_disc.clicked.connect(self.discover_wireless_devices)
        row_dev.addWidget(btn_disc)
        layout_wifi.addLayout(row_dev)

        row_key = QHBoxLayout()
        row_key.addWidget(QLabel("Access Key:"))
        self.txt_access_key = QLineEdit(self.wireless_access_key)
        self.txt_access_key.setEchoMode(QLineEdit.Password)
        row_key.addWidget(self.txt_access_key, stretch=1)
        btn_usb = QPushButton("USB Provision")
        btn_usb.setStyleSheet("background-color: #16476A; border: 1px solid #3B9797;")
        btn_usb.clicked.connect(self.open_usb_provision_dialog)
        row_key.addWidget(btn_usb)
        layout_wifi.addLayout(row_key)

        self.tab_medium.addTab(tab_wifi, "Wireless Wi-Fi (ESP32-S3)")

        # Tab 2: Wired Serial (COM Port)
        tab_wired = QWidget()
        layout_wired = QVBoxLayout(tab_wired)
        row_serial = QHBoxLayout()
        row_serial.addWidget(QLabel("Serial Port:"))
        self.combo_ports = QComboBox()
        row_serial.addWidget(self.combo_ports, stretch=1)
        btn_ref = QPushButton("Refresh")
        btn_ref.clicked.connect(self.refresh_ports)
        row_serial.addWidget(btn_ref)
        layout_wired.addLayout(row_serial)

        row_baud = QHBoxLayout()
        row_baud.addWidget(QLabel("Baud Rate:"))
        self.combo_baud = QComboBox()
        self.combo_baud.addItems(["921600", "115200", "57600", "9600"])
        row_baud.addWidget(self.combo_baud, stretch=1)
        layout_wired.addLayout(row_baud)

        self.tab_medium.addTab(tab_wired, "Wired Serial (USB/COM)")

        # Tab 3: Simulator
        tab_sim = QWidget()
        layout_sim = QVBoxLayout(tab_sim)
        lbl_sim_info = QLabel("Connect to BioWave EMG Simulator app running locally on <b>socket://127.0.0.1:7000</b>")
        lbl_sim_info.setWordWrap(True)
        lbl_sim_info.setStyleSheet("color: #A9C2CF;")
        layout_sim.addWidget(lbl_sim_info)

        self.tab_medium.addTab(tab_sim, "BioWave Simulator")

        setup_layout.addWidget(self.tab_medium)

        # Connection Control Row
        conn_row = QHBoxLayout()
        self.lbl_medium_hint = QLabel("Select medium and click Connect.")
        self.lbl_medium_hint.setStyleSheet("color: #A9C2CF;")
        conn_row.addWidget(self.lbl_medium_hint, stretch=1)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setStyleSheet("font-size: 15px; padding: 8px 18px;")
        self.btn_connect.clicked.connect(self.toggle_connection)
        conn_row.addWidget(self.btn_connect)
        setup_layout.addLayout(conn_row)

        layout.addWidget(grp_setup)

        # Initial refresh
        self.refresh_ports()

        # 2. Action Mapping Group
        grp_map = QGroupBox("2. Class to Action Mapping")
        map_layout = QVBoxLayout(grp_map)
        
        self.scroll_map = QScrollArea()
        self.scroll_map.setWidgetResizable(True)
        self.map_content = QWidget()
        self.map_form = QFormLayout(self.map_content)
        self.scroll_map.setWidget(self.map_content)
        map_layout.addWidget(self.scroll_map)
        
        lbl_hint = QLabel("<i>Load a model to view gesture classes.</i>")
        lbl_hint.setStyleSheet("color: #A9C2CF;")
        self.map_form.addRow(lbl_hint)
        layout.addWidget(grp_map, 1)

        # 3. Settings Group
        grp_settings = QGroupBox("3. Control Settings")
        set_layout = QFormLayout(grp_settings)
        
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(10.0, 99.9)
        self.spin_conf.setValue(65.0)
        self.spin_conf.setSuffix("%")
        set_layout.addRow("Minimum Confidence:", self.spin_conf)

        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(50, 3000)
        self.spin_speed.setValue(900)
        self.spin_speed.setSuffix(" px/sec")
        set_layout.addRow("Mouse Speed:", self.spin_speed)

        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(0.05, 5.0)
        self.spin_cooldown.setValue(0.6)
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
        self.btn_mouse_toggle.setStyleSheet(themed_button_style("success") + " QPushButton { font-size: 16px; padding: 12px; }")
        self.btn_mouse_toggle.setCheckable(True)
        self.btn_mouse_toggle.setEnabled(False)
        self.btn_mouse_toggle.toggled.connect(self.toggle_mouse_control)
        layout.addWidget(self.btn_mouse_toggle)
        
        lbl_safety = QLabel("<b>Safety Feature:</b> Move physical mouse to screen corner to abort!")
        lbl_safety.setAlignment(Qt.AlignCenter)
        lbl_safety.setStyleSheet("color: #FB7185; font-size: 11px;")
        layout.addWidget(lbl_safety)

        if not HAS_MOUSE_CONTROL:
            QMessageBox.warning(
                self,
                "Missing Mouse Library",
                "Install pyautogui for mouse control. On macOS, installing pyobjc also enables the faster native cursor path.",
            )

    def refresh_ports(self):
        self.combo_ports.clear()
        for p in serial.tools.list_ports.comports():
            self.combo_ports.addItem(f"{p.device} - {p.description}", p.device)

    def discover_wireless_devices(self):
        self.combo_devices.clear()
        try:
            self.discovered_devices = ControlProtocol.discover()
            for dev in self.discovered_devices:
                self.combo_devices.addItem(dev.summary, dev)
            if not self.discovered_devices:
                self.lbl_medium_hint.setText("No wireless devices auto-discovered. Type target IP manually.")
            else:
                self.lbl_medium_hint.setText(f"Found {len(self.discovered_devices)} wireless device(s).")
        except Exception as e:
            self.lbl_medium_hint.setText(f"Discovery notice: {e}")

    def open_usb_provision_dialog(self):
        dlg = USBProvisionDialog(self)
        dlg.exec_()

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select RF Model", "", "Joblib Files (*.joblib)")
        if path:
            try:
                artifact = joblib.load(path)
                model = artifact["model"]
                class_names = artifact["class_names"]
                self.window_samples = artifact.get("window_samples", 100)
                self.stride_samples = artifact.get("stride_samples", 25)
                
                # Dynamic channel detection
                num_ch = artifact.get("input_channels")
                if num_ch is None:
                    n_feats = getattr(model, "n_features_in_", 0)
                    if n_feats == 70: num_ch = 4
                    elif n_feats == 156: num_ch = 8
                    elif n_feats == 231: num_ch = 11
                    else: num_ch = 4
                self.num_channels = int(num_ch)
                
                self.data_buffer = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)
                self.baseline_offsets = np.zeros(self.num_channels, dtype=np.float32)

                self.inference_worker.load_model(model, class_names)
                self.txt_model_path.setText(path)
                self.build_mapping_ui(class_names)
                
                self.model_loaded = True
                self.check_ready_state()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{e}")

    def build_mapping_ui(self, class_names):
        while self.map_form.count():
            item = self.map_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        self.mapping_combos = []
        self.class_action_map = {}

        for cls in class_names:
            combo = QComboBox()
            combo.addItems(MOUSE_ACTIONS)
            
            clower = cls.lower()
            if "up" in clower: combo.setCurrentText("Move Up")
            elif "down" in clower: combo.setCurrentText("Move Down")
            elif "left" in clower and "click" not in clower: combo.setCurrentText("Move Left")
            elif "right" in clower and "click" not in clower: combo.setCurrentText("Move Right")
            elif "double" in clower: combo.setCurrentText("Double Click")
            elif "right" in clower and "click" in clower: combo.setCurrentText("Right Click")
            elif "click" in clower or "fist" in clower: combo.setCurrentText("Left Click")
            else: combo.setCurrentText("Ignore")

            combo.currentTextChanged.connect(lambda text, c=cls: self.update_map(c, text))
            self.update_map(cls, combo.currentText())
            
            self.map_form.addRow(f"Gesture: <b>{cls}</b>", combo)
            self.mapping_combos.append(combo)

    def update_map(self, cls, action):
        self.class_action_map[cls] = action

    def toggle_connection(self):
        if self.is_connected:
            self.disconnect_stream()
        else:
            self.connect_stream()

    def connect_stream(self):
        if not self.model_loaded:
            QMessageBox.warning(self, "Model Required", "Please load a model first.")
            return

        medium_idx = self.tab_medium.currentIndex()

        self.data_buffer = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)
        self.baseline_offsets = np.zeros(self.num_channels, dtype=np.float32)
        self.samples_since_last_pred = 0
        self.pred_history.clear()
        self.curr_vx = 0.0
        self.curr_vy = 0.0
        self.acc_x = 0.0
        self.acc_y = 0.0

        try:
            if medium_idx == 0:  # Wireless Wi-Fi
                target_dev = self.combo_devices.currentData()
                if isinstance(target_dev, DeviceInfo):
                    target_ip = target_dev.ip
                else:
                    target_ip = self.combo_devices.currentText().strip().split()[0]

                if not target_ip:
                    QMessageBox.warning(self, "Warning", "Select or type a valid device IP.")
                    return

                access_key = self.txt_access_key.text().strip() or DEFAULT_ACCESS_KEY

                # 1. Start UDP Listener worker
                self.active_worker = WirelessStreamWorker(WIFI_STREAM_PORT)
                self.active_worker.batch_received.connect(self.on_batch_received)
                self.active_worker.error_occurred.connect(lambda e: QMessageBox.warning(self, "Wireless Stream Error", e))
                self.active_worker.start()

                # 2. Send START stream command to hardware
                client_ip = get_local_ip_for_target(target_ip)
                ControlProtocol.start_stream(target_ip, access_key, client_ip, WIFI_STREAM_PORT)

                # 3. Start background Keepalive Ping Worker to keep ESP32 stream active
                self.keepalive_worker = KeepaliveWorker(target_ip, access_key)
                self.keepalive_worker.ping_failed.connect(lambda msg: print(f"Keepalive notice: {msg}"))
                self.keepalive_worker.start()

                self.current_wireless_device = (target_ip, access_key)
                self.lbl_status.setText(f"Status: Streaming Wireless ({target_ip})")

            elif medium_idx == 1:  # Wired Serial
                port = self.combo_ports.currentData()
                if not port:
                    port = self.combo_ports.currentText().split()[0]
                baud = int(self.combo_baud.currentText())

                self.active_worker = SerialWorker(port, baud, self.num_channels, batch_size=self.stride_samples)
                self.active_worker.batch_received.connect(self.on_batch_received)
                self.active_worker.error_occurred.connect(lambda e: QMessageBox.warning(self, "Serial Error", e))
                self.active_worker.start()
                self.lbl_status.setText(f"Status: Streaming Serial ({port})")

            else:  # Simulator
                sim_url = "socket://127.0.0.1:7000"
                self.active_worker = SerialWorker(sim_url, 921600, self.num_channels, batch_size=self.stride_samples)
                self.active_worker.batch_received.connect(self.on_batch_received)
                self.active_worker.error_occurred.connect(lambda e: QMessageBox.warning(self, "Simulator Error", e))
                self.active_worker.start()
                self.lbl_status.setText("Status: Streaming BioWave Simulator")

            self.is_connected = True
            self.btn_connect.setText("Disconnect")
            self.btn_connect.setStyleSheet("background-color: #BF092F; font-size: 15px; padding: 8px 18px;")
            self.lbl_status.setStyleSheet("color: #3B9797;")
            self.check_ready_state()

        except Exception as e:
            self.disconnect_stream()
            QMessageBox.critical(self, "Connection Failed", str(e))

    def disconnect_stream(self):
        if self.keepalive_worker:
            self.keepalive_worker.stop()
            self.keepalive_worker = None

        if self.current_wireless_device:
            target_ip, access_key = self.current_wireless_device
            try:
                ControlProtocol.stop_stream(target_ip, access_key)
            except Exception:
                pass
            self.current_wireless_device = None

        if self.active_worker:
            self.active_worker.stop()
            self.active_worker = None

        self.is_connected = False
        self.btn_connect.setText("Connect")
        self.btn_connect.setStyleSheet("font-size: 15px; padding: 8px 18px;")
        self.btn_mouse_toggle.setChecked(False)
        self.check_ready_state()
        self.lbl_status.setText("Status: Disconnected")
        self.lbl_status.setStyleSheet("color: #A9C2CF;")

    def check_ready_state(self):
        ready = self.model_loaded and self.is_connected
        self.btn_mouse_toggle.setEnabled(ready and HAS_MOUSE_CONTROL)
        if not ready and self.btn_mouse_toggle.isChecked():
            self.btn_mouse_toggle.setChecked(False)

    def toggle_mouse_control(self, checked):
        self.mouse_control_active = checked and HAS_MOUSE_CONTROL
        if checked:
            self.btn_mouse_toggle.setText("STOP MOUSE CONTROL")
            self.btn_mouse_toggle.setStyleSheet(themed_button_style("danger") + " QPushButton { font-size: 16px; padding: 12px; }")
            self.curr_vx = 0.0
            self.curr_vy = 0.0
            self.acc_x = 0.0
            self.acc_y = 0.0
            self._last_motion_tick = time.monotonic()
            self.smooth_motion_timer.start()
        else:
            self.btn_mouse_toggle.setText("ENABLE MOUSE CONTROL")
            self.btn_mouse_toggle.setStyleSheet(themed_button_style("success") + " QPushButton { font-size: 16px; padding: 12px; }")
            self.smooth_motion_timer.stop()
            self.current_active_action = "Ignore"

    def on_batch_received(self, payload):
        if self.data_buffer is None:
            return

        if isinstance(payload, dict):
            batch = payload.get("batch")
        else:
            batch = payload

        if batch is None:
            return

        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[0] == 0:
            return

        # Slice channels to match model's expected channel count
        if batch.shape[1] > self.num_channels:
            batch = batch[:, :self.num_channels]
        elif batch.shape[1] < self.num_channels:
            padding = np.zeros((batch.shape[0], self.num_channels - batch.shape[1]), dtype=np.float32)
            batch = np.hstack([batch, padding])

        # Adaptive baseline centering. Vectorising this keeps the GUI thread
        # free for the high-frequency mouse timer as sample batches arrive.
        self.baseline_offsets *= 0.95
        self.baseline_offsets += 0.05 * np.mean(batch, axis=0, dtype=np.float32)
        
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
        
        self.pred_history.append((label, conf_pct))

        labels = [l for l, c in self.pred_history if c >= req_conf]
        if labels:
            vote_label = max(set(labels), key=labels.count)
        else:
            vote_label = label

        # Update UI text
        self.lbl_prediction.setText(vote_label.upper())
        self.lbl_conf.setText(f"Conf: {conf_pct:.1f}%")
        
        if conf_pct < req_conf or self.class_action_map.get(vote_label) == "Ignore":
            self.lbl_prediction.setStyleSheet("color: #6F8A99;")
            self.current_active_action = "Ignore"
            return
            
        self.lbl_prediction.setStyleSheet("color: #3B9797;")
        action = self.class_action_map.get(vote_label, "Ignore")
        self.current_active_action = action
        self.current_active_conf = conf_pct

        # Process Discrete Mouse Clicks on gesture activation (rising edge trigger)
        if self.mouse_control_active and "Click" in action:
            if action != self.last_clicked_action:
                self.execute_mouse_click(action)
                self.last_clicked_action = action
        else:
            self.last_clicked_action = ""

    def execute_mouse_click(self, action):
        now = time.time()
        cooldown = self.spin_cooldown.value()
        if now - self.last_click_time < cooldown:
            return

        try:
            if action == "Left Click":
                self.mouse_backend.click(button="left")
            elif action == "Right Click":
                self.mouse_backend.click(button="right")
            elif action == "Double Click":
                self.mouse_backend.click(button="left", clicks=2)
            self.last_click_time = now
        except (MouseSafetyTriggered, pyautogui.FailSafeException if HAS_PYAUTOGUI else RuntimeError):
            self._disable_for_safety()

    def _disable_for_safety(self):
        self.btn_mouse_toggle.setChecked(False)
        QMessageBox.critical(self, "Safety Triggered", "Mouse hit screen corner. Control disabled for safety.")

    def update_smooth_mouse_motion(self):
        """High-frequency, time-based cursor velocity update loop."""
        if not self.mouse_control_active or not HAS_MOUSE_CONTROL:
            return

        now = time.monotonic()
        elapsed_s = min(max(now - self._last_motion_tick, 0.001), 0.050)
        self._last_motion_tick = now

        action = self.current_active_action
        base_speed = float(self.spin_speed.value())

        # Determine target velocity
        target_vx, target_vy = 0.0, 0.0
        if action == "Move Up":
            target_vy = -base_speed
        elif action == "Move Down":
            target_vy = base_speed
        elif action == "Move Left":
            target_vx = -base_speed
        elif action == "Move Right":
            target_vx = base_speed

        # Exponential velocity interpolation (acceleration / deceleration smoothing)
        self.curr_vx = self.smooth_alpha * target_vx + (1.0 - self.smooth_alpha) * self.curr_vx
        self.curr_vy = self.smooth_alpha * target_vy + (1.0 - self.smooth_alpha) * self.curr_vy

        # Sub-pixel accumulation. curr_v* is pixels/second, not pixels/tick.
        self.acc_x += self.curr_vx * elapsed_s
        self.acc_y += self.curr_vy * elapsed_s

        move_x = int(self.acc_x)
        move_y = int(self.acc_y)

        if move_x != 0 or move_y != 0:
            self.acc_x -= move_x
            self.acc_y -= move_y
            try:
                self.mouse_backend.move_relative(move_x, move_y)
            except (MouseSafetyTriggered, pyautogui.FailSafeException if HAS_PYAUTOGUI else RuntimeError):
                self._disable_for_safety()

    def closeEvent(self, event):
        self.disconnect_stream()
        if self.inference_worker:
            self.inference_worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_theme(app, font_size=15)
    window = MouseControllerApp()
    window.show()
    sys.exit(app.exec_())
