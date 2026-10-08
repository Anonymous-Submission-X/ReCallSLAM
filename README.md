# ReCall-SLAM website

Anonymous static research website. No app server or remote 3D dependency is
required:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. The point clouds load only when their section
enters the viewport. Drag in either 3D pane to rotate both; scroll to zoom.

## Rebuild display assets

The sibling FwdSlam checkout and consolidated benchmark results must be mounted.
Run from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/build_gallery_3d.py
PYTHONDONTWRITEBYTECODE=1 python scripts/build_growth.py
PYTHONDONTWRITEBYTECODE=1 python scripts/build_pipeline.py
```

The first command samples delivered point clouds. The second reprojects
selected long-sequence RGB-D frames and writes 3D packs, H.264 growth videos,
and posters. It requires NumPy, OpenCV, Pillow and `imageio-ffmpeg`.
The last command renders the paper's pipeline PDF using PyMuPDF.
See [ASSET_STATUS.md](ASSET_STATUS.md) for source coverage and display rules.

The vendored three.js module and OrbitControls are MIT licensed; see
`vendor/THREE_LICENSE.txt`.
