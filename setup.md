# Setup

Getting this app running on a fresh Ubuntu/Debian machine involves three
separate kinds of dependencies. Only the first two are handled by `pip`/`apt`
— the third is a vendor SDK that has to be installed by hand.

## 1. Python packages

```
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r requirements-realsense.txt   # only if using --backend realsense
```

**Create the venv from a real system Python, not a conda one.** If `python3`
(or whatever interpreter `python3 -m venv` is invoked with) is itself a conda
install - `readlink -f venv/bin/python3` shows a path under something like
`~/miniconda3/` - the venv's `python` binary is a symlink to that conda
interpreter, not an independent copy. That conda binary carries its own
`DT_RPATH` pointing at conda's `lib/` directory, and old-style `DT_RPATH` is
searched *before* `LD_LIBRARY_PATH` or the system's `ld.so.cache` for every
native library the process loads - not just conda's own dependencies. This
was diagnosed the hard way: PySide6's Qt ended up silently loading conda's
`libstdc++.so.6` (an older ABI than the system's) and several other libraries
in place of the system copies the whole native GUI/SDK stack was actually
built against, producing exactly the kind of intermittent, hard-to-place
memory corruption (`free(): invalid pointer`, segfaults with no clear cause)
that shows up only once a native camera SDK is also loaded in-process.
Use an interpreter with no such rpath, e.g. `/usr/bin/python3.12`, and verify
with `readlink -f venv/bin/python3` that it does *not* resolve into a conda
directory before installing anything into it.

`requirements.txt` covers the Nion (`ids_peak`) backend, PySide6, matplotlib,
and numpy - everything needed for `--backend ids_peak` (the default).
`requirements-realsense.txt` is separate and only needed for
`--backend realsense`, since it pulls in `pyrealsense2`, which most Nion-only
setups won't want. `--backend singray` needs no pip package - see section 4.

Both files pin exact versions rather than `>=` floors, verified by installing
them into a fresh venv (per the note above) and diffing `pip freeze` against
the environment they were captured from - byte-identical. These are direct
dependencies only; their own transitive sub-dependencies (`contourpy`,
`fonttools`, `pillow`, `shiboken6`, etc.) are left unpinned, so a future
install could still drift there, though these are all lightweight, low-churn
libraries where that risk is small. Re-derive the pins the same way after a
deliberate upgrade: install fresh, confirm the app still works against real
hardware, then `pip freeze` and copy the relevant lines back in - don't bump
versions by hand without re-verifying.

Most of these wheels (PySide6, matplotlib, the `ids_peak*` packages,
`pyrealsense2`) bundle their own compiled shared libraries, so `pip` alone
gets the Python side working - but those bundled libraries still dynamically
link against system libraries at runtime, which is where apt comes in.

## 2. System (apt) packages

Determined by running `ldd` on the actual compiled artifacts the app loads
(PySide6's Qt6 libraries and `libqxcb.so` platform plugin, the `ids_peak`
wheel's bundled `.so`s, the IDS GenTL producer, `pyrealsense2`'s extension)
and mapping every resolved shared-library path back to the Debian package
that owns it via `dpkg -S`. On a desktop Ubuntu install these are usually
already present; on a minimal/server image they won't be.

```
sudo apt install \
  libatomic1 libbrotli1 libbz2-1.0 libc6 libdbus-1-3 libegl1 libexpat1 \
  libfontconfig1 libfreetype6 libgcc-s1 libgl1 libglib2.0-0t64 libglvnd0 \
  libglx0 libpcre2-8-0 libpng16-16t64 libstdc++6 libsystemd0 libudev1 \
  libusb-1.0-0 libx11-6 libx11-xcb1 libxau6 libxcb1 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render0 \
  libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-util1 \
  libxcb-xfixes0 libxcb-xkb1 libxdmcp6 libxkbcommon0 libxkbcommon-x11-0 \
  libzstd1 zlib1g
```

`libxcb-cursor0` is the one most likely to be missing on a minimal install -
without it, PySide6/Qt6's xcb platform plugin fails to load ("could not load
the Qt platform plugin xcb") and the app won't start at all.

This list was derived empirically, not from documentation - if a future
dependency bump pulls in a Qt/library version with different linkage, re-derive
it the same way (`ldd <file> | dpkg -S -` on the relevant `.so`s) rather than
trusting it to stay accurate forever. The definitive check is always trying a
fresh install on a clean container/VM and letting any
`ImportError`/plugin-load failures point at whatever's still missing.

## 3. IDS peak SDK (not an apt or pip package)

The Nion camera backend (`ids_backend.py`) needs the actual GigE Vision
GenTL producer - the low-level driver that talks to the camera - which is
**not** included in the `ids_peak` pip wheel and **not** available via `apt`.
It ships as IDS Imaging's own "IDS peak" SDK installer, downloaded from IDS's
website and installed to a fixed location (this machine has it under
`/opt/ids-peak_26.06-720_amd64/`). The backend looks for
`ids_gevgentl.cti` under the standard GenTL search path; set the
`GENICAM_GENTL64_PATH` environment variable to override where it looks (either
the `.cti` file directly or a directory to find it under) - see
`_find_gev_producer_path()` in `src/nion_app/camera/ids_backend.py`.

No equivalent step is needed for `--backend realsense`: `pyrealsense2`'s wheel
is self-contained.

## 4. Singray Stereo PRO SDK (not an apt or pip package)

The Singray backend (`singray_backend.py`) needs the vendor's `xvsdk` Python
module and its native dependency `libxvisio-CInterface-wrapper.so` - neither
is a pip package. `xvsdk.py` ships as a bare file inside the Stereo PRO SDK
download (`python/xvsdk.py`); the backend adds that directory to `sys.path`
and imports it directly. Set `SINGRAY_SDK_PYTHON_PATH` to that directory if
it isn't at the default location the backend checks
(`~/Desktop/singray/sdk/StereoPRO/python`) - see `_import_xvsdk()` in
`src/nion_app/camera/singray_backend.py`.

The SDK checkout under `~/Desktop/singray/sdk/StereoPRO/` only included
Windows binaries (`bin/*.dll`, `lib/*.lib`) - no Linux `.so`. The actual
Linux native library comes from a **separate vendor `.deb` package**, found
elsewhere in the same download tree:

```
~/Desktop/singray/sdk/SDK_20240905/For_Ubuntu/AMD64/xvsdk_3.2.0-20240905_{bionic,focal,jammy}_amd64.deb
~/Desktop/singray/sdk/SDK_20240905/For_Ubuntu/ARM64/xvsdk_3.2.0-20240905_{bionic,focal,jammy}_arm64.deb
```

Pick the file matching the target machine's Ubuntu codename (`lsb_release -cs`)
and CPU architecture - these are named for Ubuntu 18.04/20.04/22.04
specifically, not any version, so pick the closest match if the target is a
different release (this dev machine runs 26.04 "resolute" and uses the
`jammy` build successfully - `dpkg -i` doesn't enforce an exact release
match, only architecture). Install with either:

```
sudo apt install ./xvsdk_3.2.0-20240905_jammy_amd64.deb   # resolves Depends via apt
# or
sudo dpkg -i xvsdk_3.2.0-20240905_jammy_amd64.deb && sudo apt --fix-broken install
```

then `sudo ldconfig`. This installs `libxvisio-CInterface-wrapper.so`,
`libxvsdk.so`, and every `libxslam-*`/`libxvuvc`/`libsony_iu456` library this
backend's dependency chain needs, plus the `all_stream`/`demo-api`/etc. CLI
tools and CMake config - none of which are needed by this project, only the
shared libraries are. `ldconfig -p | grep xvisio` confirms whether a given
machine already has the library installed.

**Known gap - OpenCV/TBB version pinning across Ubuntu releases.** The `.deb`
declares `Depends: ..., libopencv-dev, ...` with no version pin, expecting
whatever OpenCV build is that Ubuntu release's apt default at the time the
`.deb` was built - Ubuntu 22.04 "jammy" ships OpenCV 4.5.4, packaged under
the literal soname `libopencv_core.so.4.5d` (a normal Debian packaging
detail for that specific OpenCV version, not a debug build), alongside
`libtbb.so.2`. `libxvsdk.so` is linked against those *exact* sonames. On a
target machine actually running jammy, `apt install libopencv-dev` (pulled
in automatically by the `.deb`'s own Depends) supplies them correctly with
no extra work. On **any other Ubuntu release** (like this dev machine, on
26.04, whose apt only offers OpenCV 4.10), those exact sonames aren't
available through apt at all - this dev machine currently has
`/usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5d`,
`libopencv_imgproc.so.4.5d`, and `libtbb.so.2` present only as loose files
with no discoverable apt/dpkg origin (not owned by any installed package,
not sourced from any `.deb` found anywhere under `~/Desktop/singray/`) -
someone placed them by hand during initial bring-up, and there is currently
no reproducible record of where they came from. To set this up on another
non-jammy machine, either:
- obtain `libopencv-core4.5d`, `libopencv-imgproc4.5d`, and `libtbb2` from
  Ubuntu 22.04's package archive (e.g. `apt-get download` against a pinned
  jammy source, or downloading the `.deb`s directly from
  `packages.ubuntu.com`) and extract just the `.so` files into
  `/usr/lib/x86_64-linux-gnu/` (`dpkg -x package.deb /tmp/extract` avoids a
  full system-wide install that could conflict with the release's own newer
  OpenCV/TBB packages), or
- copy the three loose files above directly from this dev machine.

No equivalent step is needed for `--backend ids_peak` or `--backend
realsense`.

**Resolved issue, kept for the record**: earlier revisions of `singray_backend.py`
called only `xv_start_tof()` before reading frames, which left the ToF stream
permanently emitting corrupt frames (`Incorrect frame recieved. All frame
counter are not the same.` / `Failed to apply depth processing: failed to
load meta data, information of the component is mismatch` on every read) and
eventually crashed the vendor's native library (`free(): invalid pointer`)
once a `PySide6.QtWidgets.QApplication` was also constructed in-process - the
crash looked Qt-related (100% reproducible with Qt loaded, never without it
over 1500+ cycles) but Qt was actually incidental; eliminating a real,
separately-found conda/venv library-conflict bug (see section 1) did not fix
it either. The actual cause, found by comparing against a known-working
reference script (`~/Desktop/singray/pythonTest/tofViewer.py`): `xv_start_tof()`
must not be called on its own - `slam_start()` and `stereo_start()` need to
be running first, even though this backend has no use for SLAM pose or
fisheye images itself. `SingrayBackend.start_acquisition()` now does this
and produces a steady stream of valid frames.

A related, still-live constraint: `xv_get_sn()` (reading the device's serial
number) reliably corrupts native state and crashes the process later
whenever called anywhere in a process that also runs the SLAM/stereo/ToF
streams - confirmed empirically, both before and after starting them. Neither
Singray backend ever calls it; `DeviceInfo.serial_number` is a fixed
placeholder instead (see `_PLACEHOLDER_SERIAL` in `singray_backend.py` and
`singray_stereo_backend.py`).

## 5. Singray stereo-depth backend - extra pip package and a calibration file

`--backend singray_stereo` needs everything section 4 covers (same device,
same native SDK), plus:

```
venv/bin/pip install -r requirements-singray-stereo.txt   # only if using --backend singray_stereo
```

This installs `opencv-python-headless` (not `opencv-python` - the `-headless`
build skips OpenCV's own GUI/video-I/O backend, which otherwise pulls in a
second copy of Qt/GTK shared libraries into the same process as PySide6 - see
the conda/venv library-conflict story in section 1 for why mixing two
copies of a native GUI toolkit in one process is exactly the kind of thing
that's caused real crashes on this project before. This backend only calls
OpenCV's numeric functions (`cv2.fisheye.*`, `cv2.remap`, `cv2.StereoSGBM_*`,
`cv2.reprojectImageTo3D`) - it never touches `cv2.imshow`/`highgui`).

It also needs a stereo calibration file - fisheye intrinsics/distortion for
each camera, the stereo pair's relative rotation/translation, and the
disparity-to-depth `Q` matrix - before its depth output means anything. This
project doesn't produce one itself; `~/Desktop/singray/pythonTest/calibrationTooling/`
does (capture checkerboard pairs with `capture.py`, compute the calibration
with `calibrate.py`, sanity-check it with `verify.py` - see that folder's own
comments for the procedure, including the checkerboard size and the FOV-guess
tuning it needs for OpenCV's fisheye optimizer to converge correctly on a
wide-angle lens). `load_configuration()` (part of the guided connection
sequence, step 4/5) auto-loads the resulting `.npz` from
`SINGRAY_STEREO_CALIBRATION_PATH` if set, else from
`~/Desktop/singray/pythonTest/calibrationTooling/stereo_calibration.npz` if
that exists - see `find_default_calibration_path()` in
`src/nion_app/camera/singray_calibration.py`. If neither is found, the guided
connection fails gracefully (same as any other connection step failing) and
recording stays disabled until a calibration is loaded via "Load Camera
Calibration File..." in the Camera Settings dock.

No equivalent step is needed for `--backend ids_peak`, `--backend
realsense`, or `--backend singray`.
