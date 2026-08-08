# What changed

## New file: `biowave_lab_suite.py`
Merges these 9 files into one, as requested:
`async_csv.py`, `performance_logger.py`, `controller_adapters.py`,
`iso_9241_9_task.py`, `experiment_manager.py`, `performance_analysis.py`,
`publication_figures.py`, `performance_figures.py`, `data_tools_ui.py`.

**Delete those 9 files from your project folder** - everything they did now
lives in `biowave_lab_suite.py`, exposed as one tabbed window:

- **Experiment Manager** - run one matched controller condition (Standard
  mouse / EMG mouse / Fixed-step / Improved) and auto-save its session.
- **ISO 9241-9 Task** - launch the bare tapping task without an experiment
  session.
- **Performance Analysis** - load logs, compute Fitts'-law metrics into a
  table, export to CSV.
- **Publication Figures** - Matplotlib figures rendered live in the window
  (not just saved blind); pick a figure from a dropdown to preview it, then
  export one or all as PNG+PDF.
- **Session Comparison** - scans every completed session under a results
  folder and builds by-session / by-participant / overall comparison
  tables, viewable and exportable to CSV.

Every tab is wrapped in a `QScrollArea`, so shrinking the window on any
MacBook display ratio/scale reflows the content instead of hiding controls
off-screen.

CSV parsing and metric computation (the only real CPU cost) run on a
background `QThread` via a small reusable `BackgroundJob` class, so the
window never freezes while a large log file is analyzed. Figure *drawing*
stays on the GUI thread, because Qt widgets/canvases must be touched from
the main thread only - this is the correct, not merely convenient, split.

## Updated: `Mouse_wireless_v2.py`
- Now imports `PerformanceLogger` from `biowave_lab_suite` instead of the
  old standalone `performance_logger.py`.
- Theming switched from a hardcoded `'Segoe UI'` (Windows-only) stylesheet
  to `app_theme.app_stylesheet()`, which already picks a native font per OS
  and is a no-op dark-titlebar call on macOS/Linux.
- The whole window is now wrapped in a resizable `QScrollArea`, and the
  window has a sensible `setMinimumSize()` instead of only a fixed initial
  `resize()`. Result: resizing to any MacBook ratio scrolls instead of
  clipping - no more hidden buttons/fields.
- Added an **"Experiments && Analysis"** panel with an **"Open Lab Suite"**
  button that launches `AnalysisSuiteWindow`, wired to the running
  controller (`emg_controller=self`) and its live `performance_logger`, so
  "EMG mouse" / "Fixed-step controller" experiment conditions work against
  your actual live connection.

## Unchanged
- `app_theme.py`, `rf_features.py`, `train_rf_model_gui.py`, `main.py`,
  `emg_simulator_app.py` are untouched. `main.py` and `emg_simulator_app.py`
  are large, separate applications outside the 9 files you asked to merge -
  say the word if you want a follow-up pass on those too.

## Install / run
```bash
python3 -m pip install PyQt5 numpy joblib pyserial pyautogui pandas matplotlib
python3 Mouse_wireless_v2.py
```
On macOS, grant **Accessibility** permission to the terminal/interpreter
running this (System Settings -> Privacy & Security -> Accessibility), or
mouse control won't move the cursor.
