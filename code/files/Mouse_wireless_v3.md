# Mouse_wireless_standalone_v3

`Mouse_wireless_standalone_v3.py` is a standalone BioWave EMG gesture-to-mouse
controller **and research instrument**. It receives EMG data from the same
wired or wireless BioWave device used by `main.py` and `main3.py`, applies
calibration and baseline correction, runs a pretrained Random Forest model,
smooths its predictions to suppress single-window misclassifications and
gesture-boundary jitter, and maps the result to computer-mouse actions.

It does not import `main.py`, `main3.py`, `mouse2.py`, or `rf_features.py` -
the EMG feature extractor is embedded in the file. It does import
`PerformanceLogger`, `PerformanceTarget`, and `AnalysisSuiteWindow` from
`biowave_lab_suite.py`, which supplies the experiment logging, ISO 9241-9
evaluation task, and comparison/export tooling described below.

## What it does

```text
BioWave EMG device
        v
Wired serial/TCP or wireless UDP stream
        v
REST/FLEX calibration and baseline correction
        v
Rolling EMG window
        v
Embedded RF feature extraction
        v
Pretrained Random Forest prediction (raw, per window)
        v
Majority-vote action buffer  <-- new in v3
        v
Gesture-to-action mapping
        v
Mouse movement or click through PyAutoGUI
```

Supported actions are:

- Ignore
- Move Up, Down, Left, Right
- Left Click
- Right Click
- Double Click

## Responsiveness, jitter, and misclassification handling (new in v3)

Every prediction the classifier makes on a single window is still shown live
on screen the instant it happens - the on-screen label and confidence never
lag behind the raw classifier. What changed is what's allowed to reach the
mouse:

- The last **5** raw predictions are kept in a rolling buffer.
- A gesture only fires a mouse action once at least **3 of those 5** agree on
  the same label (majority vote) *and* the current window's confidence clears
  **Minimum Confidence**.
- The buffer is reset automatically after calibration and after loading a new
  model, so stale votes from before a baseline change or from a different
  model's class set can never leak into a decision.

This targets two distinct problems separately:

| Problem | Symptom without buffering | How the buffer fixes it |
|---|---|---|
| Single-window misclassification | One noisy 200 ms window fires the wrong click/move | Needs 3-of-5 agreement, so one bad window is outvoted |
| Gesture-boundary jitter | Cursor stutters back and forth as the classifier flickers between two adjacent gestures | A flickering signal never reaches a stable majority, so no action fires until it settles |

The trade-off is a small amount of added latency (up to ~4 prediction
strides, roughly 200 ms with default settings) before the *first* action
after switching gestures. If a study protocol calls for a different
latency/stability balance, `RF_LABEL_SMOOTH_WINDOW` and
`RF_LABEL_SMOOTH_MIN_AGREEMENT` near the top of the file are the two knobs to
change together.

**Minimum Confidence**, **Mouse Speed**, and **Click Cooldown** in the UI
still apply on top of this buffer - they are unchanged and compose with it.

## Requirements

Install the required Python packages in the environment used to run the
application:

```bash
python3 -m pip install PyQt5 numpy joblib pyserial pyautogui pandas matplotlib
```

`pandas` and `matplotlib` are required because the **Open Lab Suite** button
launches `AnalysisSuiteWindow` from `biowave_lab_suite.py`.

On macOS, grant **Accessibility** permission to the terminal application or
Python interpreter that launches the controller. Without this permission,
EMG recognition can run but mouse movement and clicks cannot be performed.

The controller also requires a compatible pretrained Random Forest `.joblib`
model. Train this model from BioWave recordings using the integrated trainer
in `main3.py` or the project training workflow.

## Start the controller

From the `code` directory, run:

```bash
python3 Mouse_wireless_standalone_v3.py
```

The application must remain open while mouse control is active.

## Wired-device setup

Use the **Wired (USB / Serial)** tab when the EMG device sends ordinary text
samples over USB serial.

The expected stream format is one sample per line, with numeric values
separated by commas or spaces:

```text
1980.2,1975.1,2010.6,1990.0
```

Default wired settings:

| Setting | Value |
|---|---:|
| Baud rate | 921600 |
| Sample rate | 500 Hz |
| Default channels | 4 |
| Supported wired channels | 2-9 |
| Batch size | 25 samples |

Steps:

1. Connect the BioWave device and select its serial port.
2. Set **EMG Channels** to the number used by the trained model.
3. Click **Connect**.
4. Load the matching `.joblib` model.
5. Run calibration.
6. Configure gesture mappings and enable mouse control.

The built-in `socket://127.0.0.1:7000` option can be used with the project
EMG simulator.

## Wireless-device setup

Use the **Wireless (Wi-Fi)** tab for the BioWave ESP32 wireless firmware.

The controller expects the same protocol as `main3.py`:

| Function | Protocol |
|---|---|
| Device discovery | UDP broadcast on port 5001 |
| Control commands | UDP port 5001 |
| EMG stream | UDP port 5000 |
| Stream layout | 8 EMG channels + 3 IMU values |
| Packet marker | `BWIM`, version 1 |

The firmware must support `DISCOVER`, `CHALLENGE`, authenticated `START`,
`STOP`, and `PING` commands. Authentication uses the device access key and an
HMAC-SHA256 challenge response.

Steps:

1. Ensure the computer and BioWave device are on the same Wi-Fi network.
2. Enter the device access key.
3. Click **Discover** and select the detected device.
4. Click **Connect Wireless**.
5. Load a compatible model and calibrate.

For a new wireless device, use **Provision New Device (USB)** to send Wi-Fi
credentials. Provisioning expects the firmware's USB commands `INFO` and
`PROVISION|ssid|password` at 115200 baud.

## Model compatibility

The loaded model must use the same data definition as the live stream.

| Stream | Required model input |
|---|---|
| 4-channel wired EMG | 4-channel model |
| N-channel wired EMG | N-channel model |
| Wireless BioWave EMG | 8-channel EMG model |

For wireless operation, the classifier uses the first eight channels, which
are EMG values. The three IMU values are retained in the stream buffer but
are not part of a standard 8-channel EMG model.

The model artifact should contain at least:

```text
model
class_names
window_samples
stride_samples
input_channels
sample_rate
```

The embedded feature extractor must match the one used during training. It
generates 15 features per channel, one RMS-ratio feature per channel, and
pairwise channel-correlation features. A four-channel model therefore uses
70 features.

## Calibration

Calibration is required after connecting and loading a model.

1. Click **Calibrate**.
2. During **REST**, keep the target muscle relaxed and still.
3. During **FLEX**, perform a stable muscle contraction.
4. Wait for the completion message.

The controller uses the median REST value of each EMG channel as the
baseline offset. It subtracts this offset from incoming samples and slowly
adjusts it during near-rest periods to compensate for drift. Completing
calibration also resets the majority-vote action buffer described above, so
no pre-calibration predictions can influence a post-calibration decision.

The FLEX phase is retained as a guided consistency step. In this controller
version, the REST capture is the value used directly for baseline
correction.

## Gesture mapping and control

Once a model is loaded, each class is shown with a selectable mouse action.
The controller attempts sensible defaults from class names, for example:

- names containing `up`, `down`, `left`, or `right` map to cursor movement;
- names containing `click` or `fist` map to left click;
- other names default to Ignore.

Before enabling control, set:

- **Minimum Confidence** - predictions below this percentage do not perform
  actions, even if the majority-vote buffer agrees.
- **Mouse Speed** - cursor movement distance per accepted prediction.
- **Click Cooldown** - minimum time between click actions.

Enable control only after verifying the predicted gesture labels and
mappings, and after confirming the smoothed (not just raw) label looks
stable at rest and during target gestures.

## Experiments, ISO 9241-9 evaluation, and comparison export

The **Experiments && Analysis** panel's **Open Lab Suite** button launches
`AnalysisSuiteWindow` (from `biowave_lab_suite.py`), wired to this
controller's live connection and performance logger. From there:

- **ISO 9241-9 Task** runs the standard multidirectional tapping-task
  protocol against this live controller, so "EMG mouse" / "Fixed-step
  controller" conditions are evaluated against your actual hardware, not a
  simulation.
- **Performance Analysis** computes ISO 9241-9 / Fitts'-law throughput and
  error-rate metrics from the resulting logs and exports them to CSV.
- **Session Comparison** aggregates every completed session into
  by-session, by-participant, and overall comparison tables, all shown
  in-app and each individually exportable to CSV.
- **Publication Figures** previews and exports the resulting figures as
  PNG+PDF.

Running these tools produces protocol-conformant, exportable data for
whatever certification or internal validation process your lab is
pursuing - the software implements the ISO 9241-9 evaluation protocol, it
does not itself constitute certification of the device or study.

## Safety

PyAutoGUI's fail-safe feature is enabled. Move the physical cursor to a
screen corner to trigger its emergency stop. The application then disables
mouse control.

Recommended operating practice:

1. Begin with all gesture mappings set to **Ignore**.
2. Confirm that predictions are stable - watch the raw label for a few
   seconds at rest before trusting the smoothed/actuated behavior.
3. Enable one movement action at a time.
4. Add click actions only after movement is reliable.
5. Keep the physical mouse available at all times.

## Troubleshooting

| Problem | Likely cause / resolution |
|---|---|
| No wired samples | Verify port, 921600 baud, channel count, and newline-delimited numeric data. |
| No wireless device found | Confirm both devices share Wi-Fi; allow UDP broadcast; verify firmware listens on port 5001. |
| Wireless connection fails | Check access key and the firmware's HMAC challenge protocol. |
| Calibration captures no data | Confirm the stream is connected and has the expected channel count. |
| Inference errors or no prediction | Use a model trained with the same channel count, sample rate, feature extractor, window, and stride. |
| Cursor does not move | Install `pyautogui` and grant macOS Accessibility permission. |
| Cursor feels sluggish to start moving | Expected: the majority-vote buffer adds up to ~4 strides of latency before the first action after a gesture change. Lower `RF_LABEL_SMOOTH_WINDOW` if your protocol needs faster (but less stable) response. |
| Too many clicks | Raise minimum confidence and increase click cooldown. |
| Unstable predictions / occasional wrong action | Recalibrate, collect more representative training data, increase window length or improve electrode contact; also confirm `RF_LABEL_SMOOTH_MIN_AGREEMENT` hasn't been lowered. |

## Runtime behavior

The application uses separate Qt threads for serial/wireless acquisition and
RF inference. Incoming batches update a 1,000-sample rolling buffer. At each
model stride, the newest full window is submitted for classification. The
inference worker retains only the latest pending window, preventing an
inference backlog from making the controller respond to stale EMG data. Raw
per-window predictions then pass through the majority-vote action buffer
described above before ever reaching `execute_mouse_action()`.

## Research-use disclaimer

This application is a research and evaluation tool, not a certified medical
or assistive device. Prediction confidence, the majority-vote buffer, and
all logged metrics are provided to support your own study design,
validation, and (if applicable) certification process - they are not a
substitute for it. Always keep the physical mouse available and start every
new calibration or model with all gesture mappings set to Ignore.

## Important limitation

This controller is designed for the BioWave device protocol already
compatible with `main.py` and `main3.py`. It is not intended as a generic
driver for arbitrary EMG hardware or arbitrary wireless packet formats.
