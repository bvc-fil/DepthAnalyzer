# Nion Depth Viewer

A PySide6 desktop application for connecting to an IDS Nion 3D time-of-flight
(ToF) depth camera over Ethernet, visualizing its output as a live 3D point
cloud, taking per-pixel noise measurements, and placing reference measurement
prisms in the scene.

## Purpose

Built against a real Nion camera (model NION10.67.G1.AF0130.7X) via the IDS
peak SDK, the app supports two main workflows:

1. **Live viewing** — guided camera connection/setup, then a continuously
   updated 3D point cloud rendered from depth frames.
2. **Noise characterization** — record every depth reading for every pixel in
   a user-selected rectangular ROI over a chosen duration, then inspect
   per-pixel mean/std-dev heatmaps, histograms, and time sequences to
   quantify sensor noise.

A third feature, **3D prisms**, lets the user drop rectangular boxes of exact
size into the point cloud (via in-scene drag handles or numeric entry) for
rough physical measurements against the live geometry.

Scene save/load (camera config + prisms + recordings, per `outlines.txt`) is
specified but not yet implemented.

## Architecture

Layered, backend-agnostic design under `src/nion_app/`:

- **`camera/`** — hardware abstraction and pure data processing, no Qt/UI
  dependency.
  - `backend.py` — `CameraBackend` ABC (discover/connect/configure/acquire)
    plus the `Frame`/`CameraParameters`/`DeviceInfo` dataclasses. Lets the
    rest of the app stay ignorant of the SDK, and a mock backend could stand
    in for hardware-less testing.
  - `ids_backend.py` — the real `IdsPeakBackend` implementation against
    `ids_peak`. Handles GenTL producer path auto-detection, GenICam node
    access (exposure/gain/frame rate, UserSet load), multipart buffer
    decoding into intensity/range/confidence images, and depth scaling.
  - `connection_flow.py` — `run_guided_connection()`, a linear 5-step
    power → physical link → connect → load config → set parameters sequence
    with a `ConnectionStepFailed` exception carrying the specific failing
    step, for UI-friendly error reporting.
  - `point_cloud.py` — pinhole back-projection of a depth image into XYZ
    points, using nominal intrinsics derived from the camera's published FOV
    (no factory per-unit calibration is available from the SDK yet).
  - `noise_recording.py` — `NoiseRecorder` (accumulates one frame at a time
    into flat per-sample arrays keyed by pixel x/y/timestamp) and
    `NoiseRecording` (post-hoc analysis: per-pixel mean/std, dropout-aware
    masking, histogram bin sharing, single-pixel time series).
- **`scene/prism.py`** — `Prism` dataclass (center/size/rotation) with Euler
  angle conversion and corner-point math, including decoding a prism back
  from a VTK box widget's output points.
- **`viewer/`** — Qt/VTK (pyvista) widgets.
  - `point_cloud_view.py` — the core 3D viewer: wheel zoom, right-drag orbit,
    middle-drag pan, reset-view button. Left button is freed from camera
    control for prism picking.
  - `prism_scene_view.py` — extends the point cloud view with prism
    rendering, click-to-select, an interactive `vtkBoxWidget` for
    dragging position/size/rotation, and Delete/Backspace removal.
  - `ir_roi_view.py` — displays the live IR/intensity image and lets the
    user drag out a rectangular ROI for noise recording.
- **`app/`** — top-level Qt widgets wiring everything together.
  - `main_window.py` — `MainWindow`: connects on startup, runs a shared
    ~10 Hz frame-acquisition timer that feeds both the 3D viewer and the
    noise panel, and handles careful teardown of native VTK/IDS-peak
    resources on close.
  - `prism_panel.py` — numeric position/size/rotation entry + add/delete
    buttons, kept in sync with the in-scene box widget.
  - `noise_panel.py` — duration picker, ROI status, record/progress
    controls, and "Show Results" launcher.
  - `noise_results_view.py` — matplotlib-based results window: per-pixel
    mean/std/IR heatmaps (click to inspect a pixel), adjustable std-dev
    color gradient, per-pixel histogram and time-sequence plots, plus a
    recording-wide statistics tab (mean-vs-std scatter, aggregate
    histograms).
- **`qt_bootstrap.py`** — forces `QT_QPA_PLATFORM=xcb` under Wayland before
  PySide6 is imported, since VTK's GL context creation fails under native
  Wayland on the dev machine.
- **`logging_setup.py`** — rotating file handler + stdout, configured once
  at startup; every camera/UI action is logged per spec.

## Entry points

- `scripts/run_app.py` — launches the full app against the real
  `IdsPeakBackend`.
- `scripts/connect_camera.py` — headless connection smoke test.
- `scripts/test_*.py` — standalone manual/visual checks for the viewer,
  ROI outline, prisms, noise panel, and live app, generating reference
  images under `logs/`.

## Design intentions worth noting

- Strict separation between SDK-specific code (`ids_backend.py`) and
  everything else, via the `CameraBackend` interface — enables testing
  without hardware and isolates the one module that will need updates if
  IDS ever exposes a calibrated depth output.
- All noise-recording analysis lives on `NoiseRecording`/`NoiseRecorder` as
  pure numpy operations, decoupled from the Qt display code that consumes
  it.
- Every camera-affecting or lifecycle-relevant action is logged (connection
  steps, parameter changes, acquisition start/stop, resource teardown),
  reflecting the spec's requirement for a full audit trail to file and
  stdout.
- Two synchronized editing paths for prisms (in-scene widget + numeric
  fields) are treated as a first-class requirement, not an afterthought —
  `set_prism_transform()` is the single point both paths drive through.
