# Nion Noise Measurement

A PySide6 desktop application for connecting to an IDS Nion 3D time-of-flight
(ToF) depth camera over Ethernet and characterizing its per-pixel depth
noise.

## Purpose

Built against a real Nion camera (model NION10.67.G1.AF0130.7X) via the IDS
peak SDK, the app supports:

1. **Guided connection/setup** — a logged, step-by-step sequence covering
   power, physical link, control connection, configuration load, and
   gain/exposure/frame-rate setting. The app still launches and is usable
   (for reviewing previously saved datasets) even if no camera is connected.
2. **Noise characterization** — record every depth reading for every pixel in
   a user-selected rectangular ROI (dragged on the live IR image) over a
   chosen duration, then inspect per-pixel mean/std-dev heatmaps, histograms,
   time sequences and their FFT, and recording-wide statistics to quantify
   sensor noise.
3. **Saving/loading recordings** — a completed (or previously saved)
   recording — including the camera's exposure/gain/frame-rate and which
   physical sensor produced it — can be written to / read back from a single
   `.npz` file, so results can be reviewed later or on a machine with no
   camera attached at all.

A live 3D point-cloud view and 3D measurement-prism placement were part of an
earlier iteration but have been removed to keep the app focused on the
noise-measurement workflow.

## Launching the app

```
python scripts/run_app.py [--backend {ids_peak,realsense,singray}]
```

`--backend` (or the `NION_CAMERA_BACKEND` env var) picks which sensor backend
to use; it defaults to `ids_peak`, the real Nion camera over Ethernet.
`realsense` targets an Intel RealSense D455 over USB instead, and `singray`
targets a Singray Stereo PRO (also USB) via its `xvsdk` SDK.

On startup the app runs the guided connection sequence automatically. If a
camera is found, live acquisition starts immediately (IR/depth preview,
camera settings) at the camera's own configured frame rate. If no camera is
found (unplugged, unpowered, wrong backend selected, etc.), a warning dialog
explains why, and the main window still opens with recording controls
disabled — **"Load Dataset..." always works regardless**, so a previously
saved recording can still be reviewed on a machine with no sensor attached.

Other entry points:
- `scripts/connect_camera.py` — headless connection smoke test (no UI).
- `scripts/test_noise_panel.py`, `scripts/test_roi_outline.py` — standalone
  manual/visual checks for the noise-measurement workflow and ROI selection,
  generating reference images under `logs/`.

## Using the app

**Main window layout:** the central panel is the noise-measurement workflow
(live IR view + controls); a "Depth Preview" dock (right) shows a live,
false-color depth image with per-pixel readout on hover; a "Camera Settings"
dock (left) exposes live exposure/gain/frame-rate controls and calibration
file upload.

**Recording noise data** (requires a connected camera):
1. Drag a rectangle on the live IR image to select the ROI to measure.
2. Pick a recording duration (seconds) and click **Start Recording**. Every
   acquired frame's depth readings within the ROI are captured until the
   duration elapses (progress bar tracks it); the acquisition rate follows
   whatever frame rate is set in the Camera Settings dock.
3. Click **Show Results** to open the results window (the heavier analysis
   runs on a background thread so the UI stays responsive while it computes).
4. Click **Save Dataset...** to write the just-finished recording to a
   `.npz` file (defaults to a timestamped filename) — including the ROI,
   every depth/timestamp sample, the camera's exposure/gain/frame-rate at
   record time, and which camera model/serial number recorded it.

**Reviewing a saved recording** (no camera required): click **Load
Dataset...**, pick a `.npz` file, and its results window opens automatically
— its title bar shows the loaded file's name.

**Results window** — two tabs:
- *Per-Pixel Heatmaps*: mean-depth, std-dev, and IR-reference heatmaps over
  the ROI (click any of them to inspect a different pixel; the std-dev
  heatmap's color gradient and clipping percentiles are adjustable). The
  selected pixel's depth histogram, its readings in time-sequence order, and
  the FFT of that sequence (useful for spotting periodic noise sources) are
  shown below.
- *Overall Statistics*: mean-vs-std scatter (one point per pixel), aggregate
  depth and std-dev histograms across every pixel/reading, and a histogram of
  valid-reading counts per pixel (for spotting dropout-prone pixels).

**Camera Settings dock** (only active once connected): live exposure, gain,
and frame-rate spin boxes (frame-rate changes retune the app's acquisition
polling rate to match), and a calibration-file upload for sensors that need
one.

## Architecture

Layered, backend-agnostic design under `src/nion_app/`:

- **`camera/`** — hardware abstraction and pure data processing, no Qt/UI
  dependency.
  - `backend.py` — `CameraBackend` ABC (discover/connect/configure/acquire)
    plus the `Frame`/`CameraParameters`/`DeviceInfo` dataclasses, and a
    `device_info` property exposing the currently connected device's
    identity. Lets the rest of the app stay ignorant of the SDK, and
    different sensor backends can stand in behind it.
  - `ids_backend.py` — the real `IdsPeakBackend` implementation against
    `ids_peak`. Handles GenTL producer path auto-detection, GenICam node
    access (exposure/gain/frame rate, UserSet load), multipart buffer
    decoding into intensity/range/confidence images, and depth scaling.
  - `realsense_backend.py` — an Intel RealSense D455 implementation of the
    same interface (`pyrealsense2`), for accommodating a different sensor
    with similar capabilities but no hardware confidence channel and
    preset-based (not GenICam) configuration.
  - `singray_backend.py` — a Singray Stereo PRO implementation using the
    vendor's ctypes-based `xvsdk` module (a bare `.py` file shipped alongside
    the SDK, not a pip package - see `setup.md` for locating/building its
    native `.so` dependency). This device's exposed C interface has no
    simultaneous IR channel alongside its ToF depth stream, no per-pixel
    confidence, and no exposure/gain/frame-rate control at all for the ToF
    sensor - see the module's docstring for how each of those constraints is
    handled (a depth-derived grayscale image stands in for IR/ROI-selection,
    confidence is always `None`, and parameter setters are no-ops).
  - `connection_flow.py` — `run_guided_connection()`, a linear 5-step
    power → physical link → connect → load config → set parameters sequence
    with a `ConnectionStepFailed` exception carrying the specific failing
    step, for UI-friendly error reporting.
  - `noise_recording.py` — `NoiseRecorder` (accumulates one frame at a time,
    snapshotting the camera's parameters and device identity at record
    start), `NoiseRecording` (the recorded data plus `save()`/`load()`
    to/from a single `.npz` file, and post-hoc analysis: per-pixel mean/std,
    dropout-aware masking, histogram bin sharing, single-pixel time series),
    and `compute_recording_stats()` (the expensive per-pixel-histogram
    analysis, pulled out so it can run off the Qt main thread). Pixel ids and
    per-frame timestamps are stored at their natural granularity (once per
    ROI grid / once per frame) rather than repeated per sample, since only
    the depth reading itself varies per (frame, pixel) — this keeps
    long/large recordings from needlessly ballooning in memory.
- **`viewer/`**
  - `ir_roi_view.py` — displays the live IR/intensity image and lets the user
    drag out a rectangular ROI for noise recording.
  - `depth_view.py` — live false-color depth-frame preview with per-pixel
    depth readout on hover, for quick before-capture checks.
- **`app/`** — top-level Qt widgets wiring everything together.
  - `main_window.py` — `MainWindow`: runs the guided connection on startup
    (degrading gracefully to a "no camera" state rather than failing to
    launch), runs the shared frame-acquisition timer (retuned to match the
    camera's actual configured frame rate) that feeds the noise/depth panels,
    throttles the depth preview's redraw rate while a recording is in
    progress so it doesn't compete with the recorder for main-thread time,
    and handles teardown of camera/IDS-peak resources on close.
  - `camera_settings_panel.py` — live exposure/gain/frame-rate controls and
    calibration-file upload; emits a signal on frame-rate change so
    `MainWindow` can keep its acquisition-polling rate in step.
  - `noise_panel.py` — duration picker, ROI status, record/progress
    controls, "Show Results" (computes off the main thread), "Save
    Dataset..."/"Load Dataset..." (a `.npz` round trip via `NoiseRecording`,
    with a timestamped default filename on save and file-name tracking for
    the results window's title).
  - `noise_results_view.py` — matplotlib-based results window: per-pixel
    mean/std/IR heatmaps (click to inspect a pixel), adjustable std-dev color
    gradient, per-pixel histogram, time-sequence, and FFT plots, plus a
    recording-wide statistics tab (mean-vs-std scatter, aggregate depth/std
    histograms, and a valid-readings-per-pixel histogram for spotting
    dropout-prone pixels).
- **`qt_bootstrap.py`** — forces `QT_QPA_PLATFORM=xcb` under Wayland before
  PySide6 is imported, for compatibility on the dev machine.
- **`logging_setup.py`** — rotating file handler + stdout, configured once
  at startup; every camera/UI action is logged per spec.

## Design intentions worth noting

- Strict separation between SDK-specific code (`ids_backend.py`,
  `realsense_backend.py`) and everything else, via the `CameraBackend`
  interface — enables swapping sensors without touching the noise-recording
  or UI code.
- All noise-recording analysis lives on `NoiseRecording`/`NoiseRecorder` as
  pure numpy operations, decoupled from the Qt display code that consumes
  it — this is also what lets the expensive per-pixel-histogram analysis run
  on a background `QThread` without touching any Qt widgets there.
- A missing/failed camera connection degrades gracefully rather than
  blocking the app from launching: recording is simply unavailable, while
  saved-dataset review (which needs no camera) still works.
- The acquisition-polling rate is kept in step with the camera's actual
  configured frame rate (rather than a fixed poll interval), and live
  preview redraws are throttled during recording so a high configured frame
  rate is spent on samples, not on redrawing previews nobody needs at that
  rate.
- File dialogs explicitly pass `QFileDialog.Option.DontUseNativeDialog` —
  the native/portal dialog integration on the dev machine renders as an
  unusable, near-zero-size window otherwise.
- Every camera-affecting or lifecycle-relevant action is logged (connection
  steps, parameter changes, acquisition start/stop, resource teardown),
  reflecting the spec's requirement for a full audit trail to file and
  stdout.
