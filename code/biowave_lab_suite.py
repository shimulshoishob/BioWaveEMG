"""BioWave Lab Suite - merged experiment, logging, and analysis tools.

This single file replaces and merges:
    async_csv.py, performance_logger.py, controller_adapters.py,
    iso_9241_9_task.py, experiment_manager.py, performance_analysis.py,
    publication_figures.py, performance_figures.py, data_tools_ui.py

Design goals
------------
* One tabbed window. Each tab is a self-contained "option box" for one job:
  running matched controller conditions, running a bare tapping task,
  analyzing logs, generating publication figures, and comparing sessions.
* Every tab is wrapped in a QScrollArea with setWidgetResizable(True), so
  resizing the window (any macOS display scale/ratio) never clips or hides
  a control - content reflows and scrolls instead.
* No blocking work runs on the GUI thread. CSV parsing and metric
  computation (the only genuinely heavy steps) run on a background
  QThread; the GUI thread only builds/paints Matplotlib canvases and
  Qt widgets, which Qt requires to stay on the main thread.
* Cross-platform styling comes from app_theme.py (already OS-aware); no
  Windows-only fonts or title-bar calls are used here.

Run standalone:
    python biowave_lab_suite.py

Or embed from Mouse_wireless_v2.py:
    from biowave_lab_suite import AnalysisSuiteWindow
    suite = AnalysisSuiteWindow(emg_controller=self, performance_logger_instance=self.performance_logger)
    suite.show()
"""

from __future__ import annotations

import csv
import json
import logging
import math
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from PyQt5.QtCore import QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from app_theme import app_stylesheet, apply_dark_title_bar, THEME_COLORS

import matplotlib
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#D9D9D9", "grid.linewidth": 0.6,
    "axes.grid": True,
})
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ============================================================================
# 1. async_csv.py -- non-blocking CSV persistence
# ============================================================================

CsvRecord = Mapping[str, Any]
LOG = logging.getLogger("biowave.lab")


class AsyncCsvWriter:
    """Append record batches to a CSV file on one dedicated daemon thread.

    ``submit`` only copies the supplied records into a queue, avoiding disk
    I/O in UI and cursor-control paths. Calling ``stop`` drains queued data.
    """

    def __init__(self, path: str | Path, fieldnames: Sequence[str], thread_name: str,
                 max_batches: int = 256) -> None:
        self.path = Path(path)
        self.fieldnames = tuple(fieldnames)
        self._queue: queue.Queue[list[dict[str, Any]] | None] = queue.Queue(maxsize=max_batches)
        self._stopped = threading.Event()
        self.dropped_batches = 0
        self.dropped_records = 0
        self._thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        self._thread.start()

    def submit(self, records: Iterable[CsvRecord]) -> None:
        batch = [dict(record) for record in records]
        if batch and not self._stopped.is_set():
            try:
                self._queue.put_nowait(batch)
            except queue.Full:
                # Telemetry is best-effort: never stall mouse/GUI control for disk.
                self.dropped_batches += 1
                self.dropped_records += len(batch)

    def stop(self, timeout_s: float = 2.0) -> None:
        if not self._stopped.is_set():
            self._stopped.set()
            # Shutdown may block briefly to preserve already accepted records;
            # it is never called from the high-frequency control path.
            try:
                self._queue.put(None, timeout=timeout_s)
            except queue.Full:
                LOG.warning("CSV writer shutdown timed out with %d pending batches", self._queue.qsize())
                return
            self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            batch = self._queue.get()
            if batch is None:
                return
            try:
                is_new = not self.path.exists() or self.path.stat().st_size == 0
                with self.path.open("a", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=self.fieldnames, extrasaction="ignore")
                    if is_new:
                        writer.writeheader()
                    writer.writerows(batch)
            except OSError:
                LOG.exception("CSV write error: %s", self.path)


# ============================================================================
# 2. performance_logger.py -- low-latency cursor telemetry
# ============================================================================

@dataclass(frozen=True)
class PerformanceTarget:
    target_id: str
    center_x: float
    center_y: float
    radius: float


class PerformanceLogger:
    """Buffer cursor events in memory; persist completed trials asynchronously."""

    FIELDNAMES = (
        "timestamp", "cursor_x", "cursor_y", "target_id", "target_center_x",
        "target_center_y", "target_radius", "click_event", "movement_state",
    )

    def __init__(self, output_dir: str | Path = "performance_logs", max_buffer_records: int = 20_000) -> None:
        from collections import deque
        self.records: "deque[dict[str, Any]]" = deque(maxlen=max_buffer_records)
        self._target: PerformanceTarget | None = None
        self._trial_records: list[dict[str, Any]] = []
        self._writer = AsyncCsvWriter(Path(output_dir) / "performance_trials.csv", self.FIELDNAMES, "PerformanceLoggerWriter")

    def set_target(self, target_id: str, center_x: float, center_y: float, radius: float) -> None:
        self._target = PerformanceTarget(str(target_id), float(center_x), float(center_y), float(radius))
        self._trial_records = []

    def clear_target(self) -> None:
        self._target = None
        self._trial_records = []

    def notify_target_appeared(self, target_id: str, center_x: float, center_y: float, radius: float) -> None:
        self.set_target(target_id, center_x, center_y, radius)

    def log_movement(self, cursor_x: int, cursor_y: int, movement_state: str) -> None:
        self._append_event(cursor_x, cursor_y, False, movement_state)

    def log_click(self, cursor_x: int, cursor_y: int, click_event: str) -> None:
        self._append_event(cursor_x, cursor_y, click_event, click_event)

    def notify_target_selected(self, cursor_x: int, cursor_y: int) -> None:
        if self._target is None:
            return
        self._append_event(cursor_x, cursor_y, "target_selected", "target_selected")
        self._writer.submit(self._trial_records)
        self.clear_target()

    def stop(self) -> None:
        self._writer.stop()

    def _append_event(self, cursor_x: int, cursor_y: int, click_event, movement_state: str) -> None:
        target = self._target
        record: dict[str, Any] = {
            "timestamp": time.time(), "cursor_x": int(cursor_x), "cursor_y": int(cursor_y),
            "target_id": target.target_id if target else "", "target_center_x": target.center_x if target else "",
            "target_center_y": target.center_y if target else "", "target_radius": target.radius if target else "",
            "click_event": click_event, "movement_state": movement_state,
        }
        self.records.append(record)
        if target is not None:
            self._trial_records.append(record)


# ============================================================================
# 3. controller_adapters.py -- experiment-specific controller wiring
# ============================================================================

class EmgControllerProtocol(Protocol):
    performance_logger: PerformanceLogger
    mouse_control_active: bool
    spin_speed: Any


ImprovedAdapter = Callable[[EmgControllerProtocol, "ExperimentConfig"], Callable[[], None] | None]


@dataclass(frozen=True)
class ExperimentConfig:
    participant_id: str
    controller_mode: str
    base_target_width: int
    base_target_distance: int
    trials_per_block: int
    random_seed: int
    fixed_step_size: float


class ControllerSession:
    """Temporarily route a controller's telemetry and settings into one session."""

    def __init__(
        self,
        controller: EmgControllerProtocol | None,
        logger: PerformanceLogger,
        config: ExperimentConfig,
        improved_adapter: ImprovedAdapter | None = None,
    ) -> None:
        self.controller = controller
        self.logger = logger
        self.config = config
        self.improved_adapter = improved_adapter
        self._original_logger: PerformanceLogger | None = None
        self._original_speed: int | None = None
        self._cleanup: Callable[[], None] | None = None

    def start(self) -> None:
        if self.config.controller_mode == "Standard mouse":
            return
        if self.controller is None:
            raise RuntimeError("A running EMG controller is required for this condition.")
        self._original_logger = self.controller.performance_logger
        self.controller.performance_logger = self.logger
        if self.config.controller_mode == "Fixed-step controller":
            self._original_speed = int(self.controller.spin_speed.value())
            self.controller.spin_speed.setValue(int(round(self.config.fixed_step_size)))
        elif self.config.controller_mode == "Improved controller":
            if self.improved_adapter is None:
                raise RuntimeError("No improved-controller adapter was configured.")
            self._cleanup = self.improved_adapter(self.controller, self.config)

    def stop(self) -> None:
        if self.controller is not None and self._original_logger is not None:
            self.controller.performance_logger = self._original_logger
        if self.controller is not None and self._original_speed is not None:
            self.controller.spin_speed.setValue(self._original_speed)
        if self._cleanup is not None:
            self._cleanup()
        self._original_logger = None
        self._original_speed = None
        self._cleanup = None


CONTROLLER_MODES = (
    "Standard mouse",
    "EMG mouse",
    "Fixed-step controller",
    "Improved controller",
)


# ============================================================================
# 4. iso_9241_9_task.py -- ISO 9241-9 multidirectional tapping task
# ============================================================================

@dataclass(frozen=True)
class DifficultyLevel:
    name: str
    target_width: int
    target_distance: int


def build_difficulty_levels(base_width: int, base_distance: int) -> list[DifficultyLevel]:
    return [
        DifficultyLevel("Easy", round(base_width * 1.25), round(base_distance * 0.75)),
        DifficultyLevel("Medium", base_width, base_distance),
        DifficultyLevel("Hard", max(12, round(base_width * 0.75)), round(base_distance * 1.25)),
    ]


class TrialCsvWriter:
    FIELDNAMES = (
        "participant_id", "block_number", "difficulty", "trial_number", "target_id",
        "target_index", "target_width", "target_distance", "target_center_x",
        "target_center_y", "trial_start", "trial_end", "movement_time_s",
        "misses_before_success",
    )

    def __init__(self, output_dir: str | Path = "iso9241_9_logs") -> None:
        self._writer = AsyncCsvWriter(
            Path(output_dir) / "iso9241_9_trials.csv", self.FIELDNAMES, "Iso92419CsvWriter",
        )

    def submit(self, record: dict[str, Any]) -> None:
        self._writer.submit([record])

    def stop(self) -> None:
        self._writer.stop()


class Iso92419TappingTask(QWidget):
    """Full-screen, eight-target, alternating ISO 9241-9 tapping task."""

    task_finished = pyqtSignal()

    def __init__(
        self,
        levels: Sequence[DifficultyLevel],
        trials_per_block: int = 16,
        randomize_blocks: bool = True,
        performance_logger: PerformanceLogger | None = None,
        participant_id: str = "unknown",
        task_output_dir: str | Path = "iso9241_9_logs",
        random_seed: int | None = None,
        controller_mode: str = "unknown",
    ) -> None:
        super().__init__()
        import random
        if not levels:
            raise ValueError("At least one difficulty level is required.")
        self.levels = list(levels)
        self.trials_per_block = int(trials_per_block)
        self.randomize_blocks = bool(randomize_blocks)
        self.performance_logger = performance_logger
        self.participant_id = str(participant_id).strip() or "unknown"
        self.controller_mode = str(controller_mode)
        self.trial_writer = TrialCsvWriter(task_output_dir)
        self._rng = random.Random(random_seed)
        self.trial_records = []

        self._blocks = []
        self._block_index = -1
        self._trial_in_block = 0
        self._target_index = 0
        self._target_center = None
        self._target_radius = 0.0
        self._trial_start_wall_time = None
        self._trial_start_clock = None
        self._misses = 0
        self._is_finished = False

        self.setWindowTitle("ISO 9241-9 Multidirectional Tapping Task")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def start(self) -> None:
        self._blocks = list(self.levels)
        if self.randomize_blocks:
            self._rng.shuffle(self._blocks)
        self.showFullScreen()
        self._begin_next_block()

    def _begin_next_block(self) -> None:
        self._block_index += 1
        self._trial_in_block = 0
        if self._block_index >= len(self._blocks):
            self._finish_task()
            return
        self._target_index = self._rng.randrange(8)
        self._show_target(self._target_index)

    @property
    def current_level(self):
        return self._blocks[self._block_index]

    def _target_centers(self) -> list[QPoint]:
        level = self.current_level
        center = QPoint(self.width() // 2, self.height() // 2)
        orbit_radius = level.target_distance / 2.0
        return [
            QPoint(
                round(center.x() + orbit_radius * math.cos((2 * math.pi * index / 8) - math.pi / 2)),
                round(center.y() + orbit_radius * math.sin((2 * math.pi * index / 8) - math.pi / 2)),
            )
            for index in range(8)
        ]

    def _show_target(self, target_index: int) -> None:
        self._target_index = target_index
        self._target_center = self._target_centers()[target_index]
        self._target_radius = self.current_level.target_width / 2.0
        self._trial_start_wall_time = time.time()
        self._trial_start_clock = time.perf_counter()
        self._misses = 0

        if self.performance_logger is not None:
            global_center = self.mapToGlobal(self._target_center)
            target_id = self._target_id()
            self.performance_logger.notify_target_appeared(
                target_id, global_center.x(), global_center.y(), self._target_radius,
            )
        self.update()

    def _target_id(self) -> str:
        return (
            f"{self.participant_id}-block-{self._block_index + 1}-"
            f"trial-{self._trial_in_block + 1}-target-{self._target_index}"
        )

    def mousePressEvent(self, event):
        if self._is_finished or event.button() != Qt.LeftButton or self._target_center is None:
            return
        dx = event.pos().x() - self._target_center.x()
        dy = event.pos().y() - self._target_center.y()
        if (dx * dx) + (dy * dy) > self._target_radius * self._target_radius:
            self._misses += 1
            return

        end_time = time.time()
        movement_time = time.perf_counter() - self._trial_start_clock
        level = self.current_level
        global_center = self.mapToGlobal(self._target_center)
        record = {
            "participant_id": self.participant_id,
            "block_number": self._block_index + 1,
            "difficulty": level.name,
            "trial_number": self._trial_in_block + 1,
            "target_id": self._target_id(),
            "target_index": self._target_index,
            "target_width": level.target_width,
            "target_distance": level.target_distance,
            "target_center_x": global_center.x(),
            "target_center_y": global_center.y(),
            "trial_start": self._trial_start_wall_time,
            "trial_end": end_time,
            "movement_time_s": movement_time,
            "misses_before_success": self._misses,
        }
        self.trial_records.append(record)
        self.trial_writer.submit(record)

        if self.performance_logger is not None:
            global_click = event.globalPos()
            self.performance_logger.notify_target_selected(global_click.x(), global_click.y())

        self._trial_in_block += 1
        if self._trial_in_block >= self.trials_per_block:
            self._begin_next_block()
        else:
            self._show_target((self._target_index + 4) % 8)

    def mouseMoveEvent(self, event):
        if self.performance_logger is not None and self._target_center is not None:
            global_position = event.globalPos()
            self.performance_logger.log_movement(
                global_position.x(), global_position.y(), f"Pointer Move ({self.controller_mode})",
            )
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101820"))
        if self._is_finished or self._block_index < 0:
            return

        level = self.current_level
        centers = self._target_centers()
        radius = level.target_width / 2.0
        painter.setRenderHint(QPainter.Antialiasing)
        for index, center in enumerate(centers):
            active = index == self._target_index
            painter.setPen(QPen(QColor("#F7F7F7") if active else QColor("#66808E"), 3))
            painter.setBrush(QColor("#2E9F92") if active else QColor("#263B47"))
            painter.drawEllipse(
                round(center.x() - radius), round(center.y() - radius),
                round(radius * 2), round(radius * 2),
            )

        painter.setPen(QColor("#EAF2F3"))
        painter.setFont(QFont("Arial", 14))
        painter.drawText(
            24, 36,
            f"Block {self._block_index + 1}/{len(self._blocks)}  |  {level.name}  |  "
            f"Trial {self._trial_in_block + 1}/{self.trials_per_block}  |  Esc: end task",
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._finish_task()
        else:
            super().keyPressEvent(event)

    def _finish_task(self) -> None:
        if self._is_finished:
            return
        self._is_finished = True
        self.trial_writer.stop()
        self.task_finished.emit()
        self.close()

    def closeEvent(self, event):
        if not self._is_finished:
            self._finish_task()
        event.accept()


# ============================================================================
# 5. performance_analysis.py -- ISO 9241-9 metric computation
# ============================================================================

MOVEMENT_STATES = {"Move Up", "Move Down", "Move Left", "Move Right"}


def _read_csv(path, required_columns):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    frame = pd.read_csv(path)
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def _as_number(value):
    return pd.to_numeric(value, errors="coerce")


def _is_click(value):
    return str(value).strip().lower() not in {"", "false", "0", "nan", "none"}


def _count_runs(mask):
    count = 0
    previous = False
    for current in mask:
        current = bool(current)
        if current and not previous:
            count += 1
        previous = current
    return count


def _trajectory_metrics(events, start, target, radius, trial_start):
    events = events.copy()
    for column in ("timestamp", "cursor_x", "cursor_y"):
        events[column] = _as_number(events[column])
    events = events.dropna(subset=["timestamp", "cursor_x", "cursor_y"]).sort_values("timestamp")
    movement_state = events["movement_state"].astype(str)
    movement = events[
        events["movement_state"].isin(MOVEMENT_STATES)
        | movement_state.str.startswith("Pointer Move")
    ]

    click_rows = events[events["click_event"].map(_is_click)]
    selected = events[events["click_event"].astype(str) == "target_selected"]
    endpoint_row = selected.iloc[-1] if not selected.empty else (click_rows.iloc[-1] if not click_rows.empty else None)
    endpoint = (
        np.array([float(endpoint_row.cursor_x), float(endpoint_row.cursor_y)])
        if endpoint_row is not None else None
    )
    start = np.asarray(start, dtype=float) if start is not None else None
    target = np.asarray(target, dtype=float)

    empty = {
        "effective_distance": np.nan, "endpoint_error_x": np.nan,
        "path_efficiency": np.nan, "target_reentries": np.nan,
        "overshoots": np.nan, "direction_reversals": np.nan,
        "time_to_first_movement_s": np.nan, "endpoint": endpoint,
    }
    if start is None or endpoint is None:
        return empty

    axis = target - start
    amplitude = float(np.linalg.norm(axis))
    if amplitude <= 1e-9:
        return empty
    unit = axis / amplitude
    event_points = movement[["cursor_x", "cursor_y"]].to_numpy(dtype=float)
    path_points = np.vstack([start, event_points, endpoint])
    segment_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    path_length = float(segment_lengths.sum())
    direct_distance = float(np.linalg.norm(endpoint - start))
    path_efficiency = direct_distance / path_length if path_length > 1e-9 else np.nan

    inside = np.linalg.norm(path_points - target, axis=1) <= float(radius)
    entries = _count_runs(inside)
    reentries = max(0, entries - 1)

    progress = (path_points - start) @ unit
    overshoots = _count_runs(progress > (amplitude + float(radius)))
    deltas = np.diff(path_points, axis=0) @ unit
    signs = np.sign(deltas[np.abs(deltas) > 1e-6])
    direction_reversals = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

    first_movement = movement.iloc[0] if not movement.empty else None
    time_to_first_movement = (
        max(0.0, float(first_movement.timestamp) - float(trial_start))
        if first_movement is not None and pd.notna(trial_start) else np.nan
    )
    endpoint_error_x = float((endpoint - target) @ unit)
    return {
        "effective_distance": direct_distance,
        "endpoint_error_x": endpoint_error_x,
        "path_efficiency": path_efficiency,
        "target_reentries": reentries,
        "overshoots": overshoots,
        "direction_reversals": direction_reversals,
        "time_to_first_movement_s": time_to_first_movement,
        "endpoint": endpoint,
    }


def analyze_logs(
    performance_csv="performance_logs/performance_trials.csv",
    trials_csv="iso9241_9_logs/iso9241_9_trials.csv",
    return_trial_level=False,
) -> pd.DataFrame:
    """Calculate ISO 9241-9 performance metrics and return a DataFrame.

    Runs entirely on plain pandas/numpy so it is safe to call from a
    background QThread; callers should never touch Qt widgets from inside it.
    """
    trials = _read_csv(
        trials_csv,
        {"block_number", "difficulty", "trial_number", "target_id", "target_width",
         "target_distance", "target_center_x", "target_center_y", "trial_start",
         "trial_end", "movement_time_s", "misses_before_success"},
    ).copy()
    events = _read_csv(
        performance_csv,
        {"timestamp", "cursor_x", "cursor_y", "target_id", "target_center_x",
         "target_center_y", "target_radius", "click_event", "movement_state"},
    ).copy()

    trials = trials.sort_values(["block_number", "trial_number"]).reset_index(drop=True)
    if "participant_id" not in trials.columns:
        trials["participant_id"] = "unknown"
    trials["participant_id"] = trials["participant_id"].fillna("unknown").astype(str)
    events = events[events["target_id"].notna() & (events["target_id"].astype(str) != "")]
    event_groups = {target_id: frame for target_id, frame in events.groupby("target_id", sort=False)}
    per_trial = []

    for _block, block_trials in trials.groupby("block_number", sort=False):
        previous_endpoint = None
        for _, trial in block_trials.iterrows():
            target_id = str(trial.target_id)
            trial_events = event_groups.get(target_id, pd.DataFrame(columns=events.columns))
            target = (float(trial.target_center_x), float(trial.target_center_y))
            radius = float(trial.target_width) / 2.0
            metrics = _trajectory_metrics(
                trial_events, previous_endpoint, target, radius, trial.trial_start,
            )
            if metrics["endpoint"] is not None:
                previous_endpoint = metrics["endpoint"]
            per_trial.append({
                "block_number": trial.block_number,
                "participant_id": trial.participant_id,
                "difficulty": trial.difficulty,
                "target_width": float(trial.target_width),
                "target_distance": float(trial.target_distance),
                "movement_time_s": float(trial.movement_time_s),
                "misses_before_success": float(trial.misses_before_success),
                "Click Error": float(trial.misses_before_success > 0),
                **{key: value for key, value in metrics.items() if key != "endpoint"},
            })

    detail = pd.DataFrame(per_trial)
    if detail.empty:
        return pd.DataFrame()
    if return_trial_level:
        return detail

    group_columns = ["participant_id", "difficulty", "target_width", "target_distance"]
    results = []
    for condition, group in detail.groupby(group_columns, dropna=False, sort=False):
        participant_id, difficulty, width, distance = condition
        endpoint_errors = group["endpoint_error_x"].dropna()
        we = 4.133 * endpoint_errors.std(ddof=1) if len(endpoint_errors) >= 2 else np.nan
        de = group["effective_distance"].mean()
        ide = math.log2((de / we) + 1.0) if pd.notna(de) and pd.notna(we) and we > 0 else np.nan
        mt = group["movement_time_s"].mean()
        total_misses = group["misses_before_success"].sum()
        trial_count = len(group)
        results.append({
            "participant_id": participant_id,
            "difficulty": difficulty,
            "target_width": width,
            "target_distance": distance,
            "trial_count": trial_count,
            "Movement Time (MT) s": mt,
            "Effective Distance (De) px": de,
            "Effective Width (We) px": we,
            "Effective Index of Difficulty (IDe) bits": ide,
            "Throughput (TP) bits/s": ide / mt if pd.notna(ide) and pd.notna(mt) and mt > 0 else np.nan,
            "Path Efficiency": group["path_efficiency"].mean(),
            "Target Re-entry Rate": (group["target_reentries"] > 0).mean(),
            "Overshoot Count": group["overshoots"].sum(min_count=1),
            "Direction Reversal Count": group["direction_reversals"].sum(min_count=1),
            "Time to First Movement s": group["time_to_first_movement_s"].mean(),
            "Click Error Rate": total_misses / (trial_count + total_misses) if trial_count + total_misses else np.nan,
        })
    return pd.DataFrame(results)


def export_results(results: pd.DataFrame, output_csv="iso9241_9_logs/iso9241_9_analysis.csv") -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path


# ============================================================================
# 6. publication_figures.py / performance_figures.py -- Matplotlib figures
# ============================================================================

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")

METRIC_COLUMNS = [
    ("movement_time_s", "MT (s)"),
    ("effective_distance", "De (px)"),
    ("Effective Width (We) px", "We (px)"),
    ("Effective Index of Difficulty (IDe) bits", "IDe (bits)"),
    ("Throughput (TP) bits/s", "TP (bits/s)"),
    ("path_efficiency", "Path efficiency"),
    ("target_reentries", "Target re-entries"),
    ("overshoots", "Overshoots"),
    ("direction_reversals", "Direction reversals"),
    ("time_to_first_movement_s", "First movement (s)"),
    ("Click Error", "Click error"),
]

FIGURE_NAMES = (
    "mt_vs_ide", "throughput", "path_efficiency", "overshoots",
    "click_error_rate", "fatigue", "boxplots",
)


def _label_no_data(axis):
    axis.text(0.5, 0.5, "Insufficient valid data", ha="center", va="center", transform=axis.transAxes)


def trial_plot_data(performance_csv, trials_csv) -> pd.DataFrame:
    """Merge trial-level detail with condition-level summary metrics.

    Pure-pandas; safe to call from a background thread.
    """
    detail = analyze_logs(performance_csv, trials_csv, return_trial_level=True)
    summary = analyze_logs(performance_csv, trials_csv)
    if detail.empty or summary.empty:
        return detail
    keys = ["participant_id", "difficulty", "target_width", "target_distance"]
    metrics = summary[keys + [
        "Effective Width (We) px", "Effective Index of Difficulty (IDe) bits", "Throughput (TP) bits/s",
    ]]
    return detail.merge(metrics, on=keys, how="left")


def _draw_bar(fig: Figure, data: pd.DataFrame, value_column, title, ylabel, aggregation="mean"):
    fig.clf()
    axis = fig.add_subplot(111)
    valid = data.dropna(subset=[value_column]) if value_column in data.columns else pd.DataFrame()
    if valid.empty:
        _label_no_data(axis)
    else:
        values = valid.groupby("participant_id")[value_column].agg(aggregation).sort_index()
        bars = axis.bar(values.index.astype(str), values.values, color=PALETTE[0], edgecolor="#333333", linewidth=0.6)
        axis.bar_label(bars, fmt="%.2f" if aggregation != "sum" else "%.0f", padding=3, fontsize=8)
        axis.set_xlabel("Participant")
        axis.set_ylabel(ylabel)
    axis.set_title(title)
    fig.tight_layout()


def draw_figure(name: str, fig: Figure, data: pd.DataFrame) -> None:
    """Populate ``fig`` in place with the requested named figure."""
    if name == "mt_vs_ide":
        fig.clf()
        axis = fig.add_subplot(111)
        valid = data.dropna(subset=["Effective Index of Difficulty (IDe) bits", "movement_time_s"])
        if valid.empty:
            _label_no_data(axis)
        else:
            for color_index, (participant, group) in enumerate(valid.groupby("participant_id", sort=True)):
                color = PALETTE[color_index % len(PALETTE)]
                axis.scatter(group["Effective Index of Difficulty (IDe) bits"], group["movement_time_s"],
                              color=color, s=38, alpha=0.8, label=str(participant), edgecolor="white", linewidth=0.4)
            if len(valid) >= 2 and valid["Effective Index of Difficulty (IDe) bits"].nunique() >= 2:
                slope, intercept = np.polyfit(valid["Effective Index of Difficulty (IDe) bits"], valid["movement_time_s"], 1)
                x = np.linspace(valid["Effective Index of Difficulty (IDe) bits"].min(),
                                 valid["Effective Index of Difficulty (IDe) bits"].max(), 100)
                axis.plot(x, slope * x + intercept, color="#202020", linewidth=1.6,
                          label=f"MT = {intercept:.3f} + {slope:.3f} x IDe")
            axis.legend(frameon=False, title="Participant", fontsize=8)
            axis.set_xlabel("Effective Index of Difficulty, IDe (bits)")
            axis.set_ylabel("Movement Time, MT (s)")
        axis.set_title("Movement Time vs. Effective Index of Difficulty")
        fig.tight_layout()

    elif name == "throughput":
        _draw_bar(fig, data, "Throughput (TP) bits/s", "Throughput per Participant", "Throughput (bits/s)")
    elif name == "path_efficiency":
        _draw_bar(fig, data, "path_efficiency", "Path Efficiency per Participant", "Path efficiency")
    elif name == "overshoots":
        _draw_bar(fig, data, "overshoots", "Overshoot Count per Participant", "Overshoot count", aggregation="sum")
    elif name == "click_error_rate":
        click_error = data.copy()
        if "misses_before_success" in click_error.columns:
            click_error["click_error_rate"] = click_error["misses_before_success"] / (1 + click_error["misses_before_success"])
        _draw_bar(fig, click_error, "click_error_rate", "Click Error Rate per Participant", "Click error rate")

    elif name == "fatigue":
        fig.clf()
        axis = fig.add_subplot(111)
        if data.empty:
            _label_no_data(axis)
        else:
            for color_index, (participant, group) in enumerate(data.groupby("participant_id", sort=True)):
                color = PALETTE[color_index % len(PALETTE)]
                group = group.sort_values(["block_number"]).reset_index(drop=True)
                session_trial = np.arange(1, len(group) + 1)
                mt = group["movement_time_s"].to_numpy(dtype=float)
                axis.scatter(session_trial, mt, color=color, alpha=0.28, s=20)
                rolling = pd.Series(mt).rolling(window=min(5, len(group)), min_periods=1, center=True).mean()
                axis.plot(session_trial, rolling, color=color, linewidth=2.0, label=str(participant))
            axis.legend(frameon=False, title="Participant", fontsize=8)
        axis.set_title("Movement Time Across Session (Fatigue Trend)")
        axis.set_xlabel("Completed trial within session")
        axis.set_ylabel("Movement Time, MT (s)")
        fig.tight_layout()

    elif name == "boxplots":
        fig.clf()
        if data.empty:
            axis = fig.add_subplot(111)
            _label_no_data(axis)
            fig.tight_layout()
            return
        participants = sorted(data["participant_id"].astype(str).unique())
        axes = fig.subplots(3, 4)
        for axis, (column, label) in zip(axes.flat, METRIC_COLUMNS):
            if column not in data.columns:
                _label_no_data(axis)
                axis.set_title(label, fontsize=9)
                continue
            series = [data.loc[data["participant_id"].astype(str) == participant, column].dropna().to_numpy()
                      for participant in participants]
            labels = [participant for participant, values in zip(participants, series) if len(values)]
            series = [values for values in series if len(values)]
            if series:
                box = axis.boxplot(series, labels=labels, patch_artist=True, medianprops={"color": "#111111", "linewidth": 1.4})
                for patch in box["boxes"]:
                    patch.set(facecolor=PALETTE[0], alpha=0.65)
                axis.tick_params(axis="x", labelrotation=30, labelsize=7)
            else:
                _label_no_data(axis)
            axis.set_title(label, fontsize=9)
            axis.set_ylabel(label, fontsize=8)
        for axis in axes.flat[len(METRIC_COLUMNS):]:
            axis.axis("off")
        fig.suptitle("Distribution of ISO 9241-9 Performance Metrics", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        raise ValueError(f"Unknown figure name: {name}")


def save_figure(name: str, data: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Render ``name`` on a throwaway figure and save it as PNG + PDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = {
        "mt_vs_ide": "mt_vs_ide_regression", "throughput": "throughput_per_participant",
        "path_efficiency": "path_efficiency", "overshoots": "overshoot_count",
        "click_error_rate": "click_error_rate", "fatigue": "fatigue_over_session",
        "boxplots": "boxplots_all_metrics",
    }
    size = (14, 10) if name == "boxplots" else (7.6, 4.8)
    fig = Figure(figsize=size)
    draw_figure(name, fig, data)
    paths = {}
    for extension in ("png", "pdf"):
        path = output_dir / f"{stems[name]}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths[extension] = path
    return paths


def generate_publication_figures(
    performance_csv="performance_logs/performance_trials.csv",
    trials_csv="iso9241_9_logs/iso9241_9_trials.csv",
    output_dir="iso9241_9_logs/figures",
) -> dict[str, dict[str, Path]]:
    """Batch/CLI entry point: compute data once, save every figure."""
    data = trial_plot_data(performance_csv, trials_csv)
    if data.empty:
        raise ValueError("No completed trials were available for figure generation.")
    return {name: save_figure(name, data, output_dir) for name in FIGURE_NAMES}


# ============================================================================
# 7. Shared GUI plumbing: scrollable tabs + background worker
# ============================================================================

def _scrollable(widget: QWidget) -> QScrollArea:
    """Wrap a tab's content so shrinking/growing the window never hides it."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    area.setWidget(widget)
    return area


def _path_row(parent: QWidget, field: QLineEdit, caption: str, file_filter: str = "CSV files (*.csv)",
              directory: bool = False, save: bool = False) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(field, 1)
    button = QPushButton("Browse")

    def _pick():
        if directory:
            path = QFileDialog.getExistingDirectory(parent, caption, field.text() or ".")
        elif save:
            path, _ = QFileDialog.getSaveFileName(parent, caption, field.text() or ".", file_filter)
        else:
            path, _ = QFileDialog.getOpenFileName(parent, caption, field.text() or ".", file_filter)
        if path:
            field.setText(path)

    button.clicked.connect(_pick)
    layout.addWidget(button)
    return row


class BackgroundJob(QThread):
    """Run any callable off the GUI thread and report back via signals."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)


def _fill_table(table: QTableWidget, frame: pd.DataFrame) -> None:
    table.setRowCount(len(frame))
    table.setColumnCount(len(frame.columns))
    table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
    for row_index, (_, row) in enumerate(frame.iterrows()):
        for column_index, value in enumerate(row):
            if pd.isna(value):
                display = ""
            elif isinstance(value, float):
                display = f"{value:.4f}"
            else:
                display = str(value)
            item = QTableWidgetItem(display)
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_index, column_index, item)
    table.resizeColumnsToContents()


# ============================================================================
# 8. Performance Analysis tab
# ============================================================================

class PerformanceAnalysisTab(QWidget):
    """Load task + performance CSVs, compute ISO 9241-9 metrics, export."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: pd.DataFrame | None = None
        self._job: BackgroundJob | None = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Performance Analysis</b> - compute ISO 9241-9 metrics from completed logs."))

        form = QFormLayout()
        self.performance_path = QLineEdit("performance_logs/performance_trials.csv")
        self.trials_path = QLineEdit("iso9241_9_logs/iso9241_9_trials.csv")
        self.output_path = QLineEdit("iso9241_9_logs/iso9241_9_analysis.csv")
        form.addRow("Performance CSV:", _path_row(self, self.performance_path, "Select performance CSV"))
        form.addRow("Task trials CSV:", _path_row(self, self.trials_path, "Select task trials CSV"))
        form.addRow("Results CSV:", _path_row(self, self.output_path, "Select result CSV location", save=True))
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("Analyze")
        self.run_button.clicked.connect(self.run_analysis)
        button_row.addWidget(self.run_button)
        self.export_button = QPushButton("Export Results CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results_csv)
        button_row.addWidget(self.export_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.status = QLabel("Select completed experiment CSV files and click Analyze.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.table = QTableWidget()
        self.table.setMinimumHeight(260)
        layout.addWidget(self.table, 1)

    def run_analysis(self) -> None:
        if self._job is not None and self._job.isRunning():
            return
        self.run_button.setEnabled(False)
        self.status.setText("Analyzing...")
        self._job = BackgroundJob(analyze_logs, self.performance_path.text(), self.trials_path.text())
        self._job.succeeded.connect(self._on_analysis_done)
        self._job.failed.connect(self._on_analysis_failed)
        self._job.finished.connect(lambda: self.run_button.setEnabled(True))
        self._job.start()

    def _on_analysis_done(self, results: pd.DataFrame) -> None:
        if results.empty:
            self.status.setText("Analysis failed: no completed trials were found in the selected files.")
            self.export_button.setEnabled(False)
            return
        self._results = results
        _fill_table(self.table, results)
        self.export_button.setEnabled(True)
        self.status.setText(f"Analysis complete: {len(results)} condition rows. Ready to export.")

    def _on_analysis_failed(self, message: str) -> None:
        self.status.setText(f"Analysis failed: {message}")
        QMessageBox.critical(self, "Analysis Error", message)

    def export_results_csv(self) -> None:
        if self._results is None:
            return
        try:
            saved_path = export_results(self._results, self.output_path.text())
            self.status.setText(f"Exported {len(self._results)} rows to {saved_path}")
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", str(exc))


# ============================================================================
# 9. Publication Figures tab
# ============================================================================

FIGURE_LABELS = {
    "mt_vs_ide": "Movement Time vs. IDe",
    "throughput": "Throughput per Participant",
    "path_efficiency": "Path Efficiency per Participant",
    "overshoots": "Overshoot Count per Participant",
    "click_error_rate": "Click Error Rate per Participant",
    "fatigue": "Fatigue Trend (MT over session)",
    "boxplots": "All-Metrics Distributions",
}


class FiguresTab(QWidget):
    """Compute trial data once, preview any figure, export PNG+PDF."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: pd.DataFrame | None = None
        self._job: BackgroundJob | None = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Publication Figures</b> - preview and export Matplotlib figures."))

        form = QFormLayout()
        self.performance_path = QLineEdit("performance_logs/performance_trials.csv")
        self.trials_path = QLineEdit("iso9241_9_logs/iso9241_9_trials.csv")
        self.output_dir = QLineEdit("iso9241_9_logs/figures")
        form.addRow("Performance CSV:", _path_row(self, self.performance_path, "Select performance CSV"))
        form.addRow("Task trials CSV:", _path_row(self, self.trials_path, "Select task trials CSV"))
        form.addRow("Figure folder:", _path_row(self, self.output_dir, "Select figure output directory", directory=True))
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        button_row.addWidget(self.load_button)

        button_row.addWidget(QLabel("Figure:"))
        self.figure_combo = QComboBox()
        for name in FIGURE_NAMES:
            self.figure_combo.addItem(FIGURE_LABELS[name], name)
        self.figure_combo.setEnabled(False)
        self.figure_combo.currentIndexChanged.connect(self._render_selected)
        button_row.addWidget(self.figure_combo, 1)

        self.export_one_button = QPushButton("Export This Figure")
        self.export_one_button.setEnabled(False)
        self.export_one_button.clicked.connect(self.export_current_figure)
        button_row.addWidget(self.export_one_button)

        self.export_all_button = QPushButton("Export All Figures")
        self.export_all_button.setEnabled(False)
        self.export_all_button.clicked.connect(self.export_all_figures)
        button_row.addWidget(self.export_all_button)
        layout.addLayout(button_row)

        self.status = QLabel("Select log files and click Load Data.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.figure = Figure(figsize=(7.2, 4.8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(360)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)

    def load_data(self) -> None:
        if self._job is not None and self._job.isRunning():
            return
        self.load_button.setEnabled(False)
        self.status.setText("Loading and computing trial metrics...")
        self._job = BackgroundJob(trial_plot_data, self.performance_path.text(), self.trials_path.text())
        self._job.succeeded.connect(self._on_data_loaded)
        self._job.failed.connect(self._on_data_failed)
        self._job.finished.connect(lambda: self.load_button.setEnabled(True))
        self._job.start()

    def _on_data_loaded(self, data: pd.DataFrame) -> None:
        if data.empty:
            self.status.setText("No completed trials were found in the selected files.")
            self.figure_combo.setEnabled(False)
            self.export_one_button.setEnabled(False)
            self.export_all_button.setEnabled(False)
            return
        self._data = data
        self.figure_combo.setEnabled(True)
        self.export_one_button.setEnabled(True)
        self.export_all_button.setEnabled(True)
        self.status.setText(f"Loaded {len(data)} trials. Select a figure to preview.")
        self._render_selected()

    def _on_data_failed(self, message: str) -> None:
        self.status.setText(f"Failed to load data: {message}")
        QMessageBox.critical(self, "Figure Data Error", message)

    def _render_selected(self) -> None:
        if self._data is None:
            return
        name = self.figure_combo.currentData()
        try:
            draw_figure(name, self.figure, self._data)
            self.canvas.draw_idle()
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Could not render figure: {exc}")

    def export_current_figure(self) -> None:
        if self._data is None:
            return
        name = self.figure_combo.currentData()
        try:
            paths = save_figure(name, self._data, self.output_dir.text())
            self.status.setText(f"Saved {paths['png']} and {paths['pdf']}")
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def export_all_figures(self) -> None:
        if self._data is None:
            return
        self.export_all_button.setEnabled(False)
        self.status.setText("Exporting all figures...")
        job = BackgroundJob(
            lambda: {name: save_figure(name, self._data, self.output_dir.text()) for name in FIGURE_NAMES}
        )
        job.succeeded.connect(self._on_export_all_done)
        job.failed.connect(self._on_export_all_failed)
        job.finished.connect(lambda: self.export_all_button.setEnabled(True))
        self._export_job = job  # keep a reference alive
        job.start()

    def _on_export_all_done(self, outputs: dict) -> None:
        self.status.setText(f"Exported {len(outputs)} figures (PNG + PDF) to {self.output_dir.text()}")

    def _on_export_all_failed(self, message: str) -> None:
        self.status.setText(f"Figure export failed: {message}")
        QMessageBox.critical(self, "Figure Error", message)


# ============================================================================
# 10. Session comparison tab
# ============================================================================

COMPARISON_METRIC_COLUMNS = [
    "Movement Time (MT) s", "Effective Distance (De) px", "Effective Width (We) px",
    "Effective Index of Difficulty (IDe) bits", "Throughput (TP) bits/s", "Path Efficiency",
    "Target Re-entry Rate", "Overshoot Count", "Direction Reversal Count",
    "Time to First Movement s", "Click Error Rate",
]


def build_comparison_tables(results_root: str | Path) -> dict[str, pd.DataFrame]:
    """Scan ``results_root/<participant>/<mode>/<timestamp>/analysis.csv``
    and build by-session, by-participant, and overall comparison tables.

    Pure pandas; safe to call from a background thread.
    """
    results_root = Path(results_root)
    analyses = []
    for analysis_path in results_root.glob("*/*/*/analysis.csv"):
        frame = pd.read_csv(analysis_path)
        if frame.empty:
            continue
        manifest_path = analysis_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        frame.insert(0, "controller_mode", manifest.get("controller_mode", "unknown"))
        frame.insert(1, "session_dir", str(analysis_path.parent))
        analyses.append(frame)
    if not analyses:
        return {}

    all_sessions = pd.concat(analyses, ignore_index=True)
    metric_columns = [column for column in COMPARISON_METRIC_COLUMNS if column in all_sessions.columns]
    by_participant = all_sessions.groupby(
        ["participant_id", "controller_mode"], dropna=False,
    )[metric_columns].mean().reset_index()
    overall = all_sessions.groupby("controller_mode", dropna=False)[metric_columns].mean().reset_index()
    return {
        "by_session": all_sessions,
        "by_participant": by_participant,
        "overall": overall,
    }


COMPARISON_FILE_NAMES = {
    "by_session": "comparison_table_by_session.csv",
    "by_participant": "comparison_table_by_participant.csv",
    "overall": "comparison_table_overall.csv",
}


class ComparisonTab(QWidget):
    """Aggregate every completed experiment session under one results root."""

    def __init__(self, results_root: str | Path = "experiment_results", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tables: dict[str, pd.DataFrame] = {}
        self._job: BackgroundJob | None = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Session Comparison</b> - aggregate every completed experiment session."))

        form = QFormLayout()
        self.results_root = QLineEdit(str(results_root))
        form.addRow("Results folder:", _path_row(self, self.results_root, "Select experiment results folder", directory=True))
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.load_button = QPushButton("Load Comparison Tables")
        self.load_button.clicked.connect(self.load_tables)
        button_row.addWidget(self.load_button)

        button_row.addWidget(QLabel("Table:"))
        self.table_combo = QComboBox()
        self.table_combo.addItem("By Session", "by_session")
        self.table_combo.addItem("By Participant", "by_participant")
        self.table_combo.addItem("Overall (by controller mode)", "overall")
        self.table_combo.setEnabled(False)
        self.table_combo.currentIndexChanged.connect(self._show_selected)
        button_row.addWidget(self.table_combo, 1)

        self.export_button = QPushButton("Export This Table")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_selected)
        button_row.addWidget(self.export_button)
        layout.addLayout(button_row)

        self.status = QLabel("Point at an experiment results folder and click Load.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.table = QTableWidget()
        self.table.setMinimumHeight(280)
        layout.addWidget(self.table, 1)

    def load_tables(self) -> None:
        if self._job is not None and self._job.isRunning():
            return
        self.load_button.setEnabled(False)
        self.status.setText("Scanning session folders...")
        self._job = BackgroundJob(build_comparison_tables, self.results_root.text())
        self._job.succeeded.connect(self._on_loaded)
        self._job.failed.connect(self._on_failed)
        self._job.finished.connect(lambda: self.load_button.setEnabled(True))
        self._job.start()

    def _on_loaded(self, tables: dict) -> None:
        if not tables:
            self.status.setText("No completed sessions (analysis.csv) were found under that folder.")
            self.table_combo.setEnabled(False)
            self.export_button.setEnabled(False)
            return
        self._tables = tables
        self.table_combo.setEnabled(True)
        self.export_button.setEnabled(True)
        sessions = len(tables.get("by_session", pd.DataFrame()))
        self.status.setText(f"Loaded {sessions} sessions.")
        self._show_selected()

    def _on_failed(self, message: str) -> None:
        self.status.setText(f"Failed to load comparison tables: {message}")
        QMessageBox.critical(self, "Comparison Error", message)

    def _show_selected(self) -> None:
        key = self.table_combo.currentData()
        frame = self._tables.get(key)
        if frame is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        _fill_table(self.table, frame)

    def export_selected(self) -> None:
        key = self.table_combo.currentData()
        frame = self._tables.get(key)
        if frame is None:
            return
        default_name = COMPARISON_FILE_NAMES.get(key, f"{key}.csv")
        default_path = str(Path(self.results_root.text()) / default_name)
        path, _ = QFileDialog.getSaveFileName(self, "Export comparison table", default_path, "CSV files (*.csv)")
        if not path:
            return
        try:
            export_results(frame, path)
            self.status.setText(f"Exported to {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def refresh(self) -> None:
        """Re-scan silently after an experiment session completes."""
        self.load_tables()


# ============================================================================
# 11. Experiment Manager tab (merged from experiment_manager.py)
# ============================================================================

class ExperimentManagerTab(QWidget):
    """Runs one reproducible controller condition and saves its session data.

    ``emg_controller`` is an optional running MouseControllerApp instance.
    ``improved_adapter`` is an optional callable receiving ``(controller, config)``
    and returning an optional cleanup callable.
    """

    def __init__(
        self,
        emg_controller: EmgControllerProtocol | None = None,
        improved_adapter: ImprovedAdapter | None = None,
        results_root: str | Path = "experiment_results",
        on_session_complete: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.emg_controller = emg_controller
        self.improved_adapter = improved_adapter
        self.results_root = Path(results_root)
        self.on_session_complete = on_session_complete
        self.active_task: Iso92419TappingTask | None = None
        self._session_logger: PerformanceLogger | None = None
        self._controller_session: ControllerSession | None = None
        self._session_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Experiment Manager</b> - run a matched Fitts' law controller condition."))
        if self.emg_controller is None:
            layout.addWidget(QLabel(
                "No live EMG controller was passed in - only 'Standard mouse' is available in this session."
            ))

        form = QFormLayout()
        self.participant_edit = QLineEdit()
        self.participant_edit.setPlaceholderText("e.g., P01")
        form.addRow("Participant ID:", self.participant_edit)
        self.controller_combo = QComboBox()
        self.controller_combo.addItems(CONTROLLER_MODES)
        form.addRow("Controller condition:", self.controller_combo)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(20, 240)
        self.width_spin.setValue(60)
        self.width_spin.setSuffix(" px")
        form.addRow("Base target width:", self.width_spin)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(100, 1000)
        self.distance_spin.setValue(400)
        self.distance_spin.setSuffix(" px")
        form.addRow("Base target distance:", self.distance_spin)
        self.trial_spin = QSpinBox()
        self.trial_spin.setRange(4, 80)
        self.trial_spin.setValue(16)
        form.addRow("Trials per block:", self.trial_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(92419)
        form.addRow("Sequence seed:", self.seed_spin)
        self.fixed_step_spin = QDoubleSpinBox()
        self.fixed_step_spin.setRange(1, 150)
        self.fixed_step_spin.setValue(30)
        self.fixed_step_spin.setSuffix(" px/action")
        form.addRow("Fixed-step size:", self.fixed_step_spin)
        results_field = QLineEdit(str(self.results_root))
        results_field.editingFinished.connect(lambda: setattr(self, "results_root", Path(results_field.text() or "experiment_results")))
        form.addRow("Results folder:", _path_row(self, results_field, "Select results folder", directory=True))
        layout.addLayout(form)

        self.run_button = QPushButton("Run Selected Condition")
        self.run_button.clicked.connect(self.run_selected_condition)
        layout.addWidget(self.run_button)

        self.status = QLabel("Configure a condition and click Run.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def _levels(self) -> list[DifficultyLevel]:
        return build_difficulty_levels(self.width_spin.value(), self.distance_spin.value())

    def _validate_controller(self, mode: str) -> bool:
        if mode == "Standard mouse":
            return True
        if self.emg_controller is None:
            QMessageBox.warning(
                self, "Controller unavailable",
                "EMG, fixed-step, and improved conditions require a running MouseControllerApp instance.",
            )
            return False
        if mode in {"EMG mouse", "Fixed-step controller"} and not self.emg_controller.mouse_control_active:
            QMessageBox.warning(
                self, "EMG control inactive",
                "Calibrate and enable mouse control in BioWave before running this controller condition.",
            )
            return False
        if mode == "Improved controller" and self.improved_adapter is None:
            QMessageBox.warning(
                self, "Improved controller unavailable",
                "No improved_adapter was configured for this session.",
            )
            return False
        return True

    def run_selected_condition(self) -> None:
        participant = self.participant_edit.text().strip()
        mode = self.controller_combo.currentText()
        if not participant:
            QMessageBox.warning(self, "Participant required", "Enter a participant ID before starting.")
            return
        if not self._validate_controller(mode):
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = mode.lower().replace(" ", "_").replace("-", "_")
        self._session_dir = self.results_root / participant / slug / timestamp
        performance_dir = self._session_dir / "performance_logs"
        task_dir = self._session_dir / "task_logs"
        self._session_dir.mkdir(parents=True, exist_ok=False)
        self._session_logger = PerformanceLogger(str(performance_dir))
        config = ExperimentConfig(
            participant, mode, self.width_spin.value(), self.distance_spin.value(),
            self.trial_spin.value(), self.seed_spin.value(), self.fixed_step_spin.value(),
        )
        (self._session_dir / "manifest.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
        self._controller_session = ControllerSession(
            self.emg_controller, self._session_logger, config, self.improved_adapter,
        )
        try:
            self._controller_session.start()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Could not start condition", str(exc))
            self._session_logger.stop()
            return

        self.active_task = Iso92419TappingTask(
            self._levels(), self.trial_spin.value(), True, self._session_logger, participant,
            str(task_dir), config.random_seed, mode,
        )
        self.active_task.task_finished.connect(self._complete_session)
        self.run_button.setEnabled(False)
        self.status.setText(f"Running: {participant} / {mode}. Task window is full-screen; press Esc to end early.")
        self.active_task.start()

    def _complete_session(self) -> None:
        try:
            self._session_logger.stop()
            performance_csv = self._session_dir / "performance_logs" / "performance_trials.csv"
            trials_csv = self._session_dir / "task_logs" / "iso9241_9_trials.csv"
            results = analyze_logs(performance_csv, trials_csv)
            export_results(results, self._session_dir / "analysis.csv")
            if self.on_session_complete is not None:
                self.on_session_complete()
            self.status.setText(f"Condition complete. Results saved in: {self._session_dir}")
            QMessageBox.information(self, "Condition complete", f"Results saved in:\n{self._session_dir}")
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Session data was saved, but analysis failed: {exc}")
            QMessageBox.warning(self, "Analysis incomplete", f"Session data was saved, but analysis failed:\n{exc}")
        finally:
            if self._controller_session is not None:
                self._controller_session.stop()
            self._controller_session = None
            self.run_button.setEnabled(True)
            self.active_task = None
            self._session_logger = None


# ============================================================================
# 12. Standalone task launcher tab (inline version of Iso92419SetupDialog)
# ============================================================================

class TaskLauncherTab(QWidget):
    """Launch a bare ISO 9241-9 task without recording an experiment session."""

    def __init__(self, performance_logger_instance: PerformanceLogger | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.performance_logger = performance_logger_instance
        self.task: Iso92419TappingTask | None = None
        self._owns_logger = performance_logger_instance is None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>ISO 9241-9 Task</b> - base width/distance generate Easy, Medium, and Hard blocks. "
            "Block order is randomized by default."
        ))
        form = QFormLayout()
        self.participant_edit = QLineEdit()
        self.participant_edit.setPlaceholderText("e.g., P01")
        form.addRow("Participant ID:", self.participant_edit)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(20, 240)
        self.width_spin.setValue(60)
        self.width_spin.setSuffix(" px")
        form.addRow("Base target width:", self.width_spin)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(100, 1000)
        self.distance_spin.setValue(400)
        self.distance_spin.setSuffix(" px")
        form.addRow("Base opposite-target distance:", self.distance_spin)
        self.trial_spin = QSpinBox()
        self.trial_spin.setRange(4, 80)
        self.trial_spin.setValue(16)
        form.addRow("Trials per block:", self.trial_spin)
        self.output_dir_edit = QLineEdit("iso9241_9_logs")
        form.addRow("Task log folder:", _path_row(self, self.output_dir_edit, "Select task log folder", directory=True))
        self.randomize_check = QCheckBox("Randomize difficulty-block order")
        self.randomize_check.setChecked(True)
        form.addRow(self.randomize_check)
        layout.addLayout(form)

        self.launch_button = QPushButton("Launch Task (full screen)")
        self.launch_button.clicked.connect(self.launch_task)
        layout.addWidget(self.launch_button)

        self.status = QLabel("Configure the task and click Launch.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def launch_task(self) -> None:
        width = self.width_spin.value()
        distance = self.distance_spin.value()
        levels = build_difficulty_levels(width, distance)
        screen = QApplication.primaryScreen().availableGeometry()
        largest_diameter = max(level.target_width for level in levels)
        largest_distance = max(level.target_distance for level in levels)
        if largest_distance + largest_diameter + 48 > min(screen.width(), screen.height()):
            QMessageBox.warning(
                self, "Targets do not fit",
                "Reduce the base target distance or width so the hard block fits on this screen.",
            )
            return
        logger = self.performance_logger or PerformanceLogger(str(Path(self.output_dir_edit.text()) / "performance_logs"))
        self.task = Iso92419TappingTask(
            levels, self.trial_spin.value(), self.randomize_check.isChecked(), logger,
            self.participant_edit.text(), self.output_dir_edit.text(),
        )
        self.task.task_finished.connect(self._on_finished)
        if self._owns_logger:
            self.task.task_finished.connect(logger.stop)
        self.launch_button.setEnabled(False)
        self.status.setText("Task running full-screen. Press Esc to end early.")
        self.task.start()

    def _on_finished(self) -> None:
        self.launch_button.setEnabled(True)
        self.status.setText(f"Task finished. {len(self.task.trial_records) if self.task else 0} trials recorded.")


# ============================================================================
# 13. Main suite window
# ============================================================================

class AnalysisSuiteWindow(QMainWindow):
    """Tabbed window hosting every experiment/analysis tool.

    Each tab is its own "option box" for one management job; every tab is
    wrapped in a QScrollArea, so resizing the window on any macOS display
    ratio reflows content instead of clipping it.
    """

    def __init__(
        self,
        emg_controller: EmgControllerProtocol | None = None,
        improved_adapter: ImprovedAdapter | None = None,
        performance_logger_instance: PerformanceLogger | None = None,
        results_root: str | Path = "experiment_results",
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("BioWave Lab Suite - Experiments & Analysis")
        self.setStyleSheet(app_stylesheet(13))
        apply_dark_title_bar(self)
        self.resize(1180, 780)
        self.setMinimumSize(760, 560)

        self.comparison_tab = ComparisonTab(results_root)
        self.experiment_tab = ExperimentManagerTab(
            emg_controller, improved_adapter, results_root,
            on_session_complete=self.comparison_tab.refresh,
        )
        self.task_tab = TaskLauncherTab(performance_logger_instance)
        self.analysis_tab = PerformanceAnalysisTab()
        self.figures_tab = FiguresTab()

        tabs = QTabWidget()
        tabs.addTab(_scrollable(self.experiment_tab), "Experiment Manager")
        tabs.addTab(_scrollable(self.task_tab), "ISO 9241-9 Task")
        tabs.addTab(_scrollable(self.analysis_tab), "Performance Analysis")
        tabs.addTab(_scrollable(self.figures_tab), "Publication Figures")
        tabs.addTab(_scrollable(self.comparison_tab), "Session Comparison")
        self.setCentralWidget(tabs)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = AnalysisSuiteWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
