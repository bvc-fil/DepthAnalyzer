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

`requirements.txt` covers the Nion (`ids_peak`) backend, PySide6, matplotlib,
and numpy - everything needed for `--backend ids_peak` (the default).
`requirements-realsense.txt` is separate and only needed for
`--backend realsense`, since it pulls in `pyrealsense2`, which most Nion-only
setups won't want.

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
