# What changed

## v3 update: `Mouse_wireless_standalone_v3.py` - responsiveness & misclassification handling

The standalone controller already collected a rolling history of raw
per-window predictions (`rf_label_history`, sized by `RF_LABEL_SMOOTH_WINDOW`)
but never actually read from it - every accepted prediction, even a single
noisy window, went straight to `execute_mouse_action()`. That's the source of
two related problems: an isolated misclassified window firing the wrong
gesture, and visible cursor jitter when the classifier flickers between two
similar gestures near a decision boundary.

**Fix - majority-vote action buffer:**
- Every raw prediction is still shown live in the UI immediately (`lbl_prediction`,
  `lbl_conf`), so the operator always sees exactly what the classifier is doing,
  including noise. Nothing about visible feedback got slower.
- Before an action is executed, the last `RF_LABEL_SMOOTH_WINDOW` (5) raw
  predictions are majority-voted. An action only fires once at least
  `RF_LABEL_SMOOTH_MIN_AGREEMENT` (60%, i.e. 3 of 5) of that buffer agrees on
  the same label *and* the current-window confidence clears **Minimum
  Confidence**.
- The buffer is cleared on calibration completion and on model load, so
  predictions from before a baseline reset or from a previous model's class
  set can never contaminate a vote.
- Net effect on responsiveness: worst case adds up to `4 x stride` of extra
  latency (with default 25-sample stride at 500 Hz, ~200 ms) before the
  *first* action after a gesture change, in exchange for suppressing
  single-window misfires and flicker. This is a standard debounce trade-off
  for EMG-classifier-driven control and is tunable via `RF_LABEL_SMOOTH_WINDOW`
  / `RF_LABEL_SMOOTH_MIN_AGREEMENT` if a study protocol needs a different
  balance of latency vs. stability.

No new dependencies, no UI changes required - this only changes *when* an
already-correct prediction pipeline is allowed to act.

## Research-tool framing

Both `Mouse_wireless_standalone_v3.py` and `biowave_lab_suite.py` are lab /
research instruments, not a consumer product:
- Predictions, baselines, and mouse actions are only as good as calibration
  quality and electrode placement for a given session - treat "Minimum
  Confidence" and the new majority-vote buffer as tunable study parameters,
  not fixed guarantees against misclassification.
- Every session's raw logs, computed metrics, and comparison tables are
  written to disk in open CSV format specifically so results can be
  re-analyzed, audited, or fed into an external stats pipeline - see below.

## ISO 9241-9 note (what "ISO" does and doesn't mean here)

`biowave_lab_suite.py`'s **ISO 9241-9 Task** and **Performance Analysis**
tabs implement the ISO 9241-9 (now folded into ISO 9241-411) multidirectional
tapping-task protocol and its standard Fitts'-law throughput/error metrics -
this is a *protocol implementation*, which is what makes the resulting
numbers directly comparable across sessions, participants, and controller
conditions. Running the software does not itself grant "ISO certification":
certification is a claim about a lab's documented process, not about a
script. Use these tabs to generate protocol-conformant data for whatever
certification or internal validation process your lab is pursuing; the
`analysis.csv` / comparison-table outputs are structured to be handed
directly to that process.

## Comparison results: already viewable and exportable

Confirming existing behavior (no change needed, documented here for
visibility): the **Session Comparison** tab in `biowave_lab_suite.py` scans
every completed session under a results folder and builds three tables -
by-session, by-participant, and overall - all shown in-app in a
`QTableWidget` and each individually exportable to CSV via **Export This
Table** (`build_comparison_tables()`, `COMPARISON_FILE_NAMES`). The
**Performance Analysis** and **Publication Figures** tabs likewise export
their own CSV / PNG+PDF outputs. Nothing here is export-blocked or
preview-only.

## New file (unchanged from prior pass): `biowave_lab_suite.py`
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

## Renamed: `Mouse_wireless_v2.py` -> `Mouse_wireless_standalone_v3.py`
- Now imports `PerformanceLogger`, `PerformanceTarget`, and
  `AnalysisSuiteWindow` from `biowave_lab_suite` instead of the old
  standalone `performance_logger.py`.
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
- **New in v3:** majority-vote action buffer (see above) so single-window
  misclassifications and gesture-boundary jitter no longer reach the mouse.

## Unchanged
- `app_theme.py`, `rf_features.py`, `train_rf_model_gui.py`, `main.py`,
  `emg_simulator_app.py` are untouched. `main.py` and `emg_simulator_app.py`
  are large, separate applications outside the 9 files originally merged -
  say the word if you want a follow-up pass on those too.

## Install / run
```bash
python3 -m pip install PyQt5 numpy joblib pyserial pyautogui pandas matplotlib
python3 Mouse_wireless_standalone_v3.py
```
On macOS, grant **Accessibility** permission to the terminal/interpreter
running this (System Settings -> Privacy & Security -> Accessibility), or
mouse control won't move the cursor.
