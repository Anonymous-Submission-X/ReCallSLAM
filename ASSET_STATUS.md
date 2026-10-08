# ReCall-SLAM website data

The public website is anonymous. Its manifests contain scene and method IDs,
never workstation paths or author metadata. All point clouds shown by the 3D
viewer are real delivered RGB-D reconstruction products.

## Included

| Section | 3D point clouds | Growth videos | Trajectory display |
| --- | --- | --- | --- |
| Multi-floor corridor, 4,300 frames | ReCall-SLAM + 7 baselines | ReCall-SLAM + 7 baselines | In the videos and 3D viewers |
| Classrooms, 27,002 frames | ReCall-SLAM + 5 baselines | ReCall-SLAM + 5 baselines | In the videos and 3D viewers |
| Tiered lecture hall, 30,079 frames | ReCall-SLAM + 5 baselines | ReCall-SLAM + 5 baselines | In the videos and 3D viewers |
| FastCaMo-Real, 12 scenes | ReCall-SLAM + 7 baselines per scene | — | Hidden: no GT trajectory |
| Oxford Spires, 4 complete scenes | ReCall-SLAM + 5 baselines per scene | — | Top-view GT vs method, plus 3D viewer trajectories |

The four Oxford routes are Christ Church 02, Keble College 04, Observatory
Quarter 01, and Blenheim Palace 01. They are complete continuous-depth captures
and have the largest combined path length and turning extent within their four
locations. The 24 depth-gap fragments are not used here.

## Oxford source coverage

All five baselines are available for the four displayed complete routes.
LingBot-Map and ABot-Recon source `pointcloud.ply` and `trajectory.tum` files
come from `../benchmark_results/oxford_spires_lingbot_abot/<scene>/<method>/`;
the other three baselines come from the consolidated benchmark directory.
The eight LingBot-Map and ABot-Recon interactive clouds and their GT-aligned
top-view trajectories are included in the website assets.

## Display and coverage rules

- Long-sequence 3D assets and videos reproject the same sampled input RGB-D
  pixels at each delivered method trajectory. Depth comes from the run's bound
  `pred_depth` input, not the separate raw `depth` folder. It is restricted to
  **0.1–8 m**. A twice-eroded valid 3×3 neighborhood and
  `range <= 0.005 m + 0.008 × depth` reject smoothed depth edges. Above-camera
  horizontal surfaces and points over 0.40 m above the camera are removed to
  avoid ceilings. Video points use one-pixel splats and a normal near-surface
  z-buffer; no farthest-point rendering rule is used. The videos use a white
  background matching the page and a bright viridis time-colored
  trajectory without an outline.
- Every 3D result estimates display up from its own camera orientations. Point
  coordinates keep metric scale. 3D pack files contain sampled RGB point data;
  the long-sequence growth videos use the larger filtered point sets.
- ScaRF-SLAM returns **middle intervals**, not complete prefixes:
  corridor frames 2,205–2,961 (756 poses), classroom frames 2,503–7,718
  (5,212 poses). VGGT-SLAM 2.0-Pi3X's lecture-hall output stops at frame
  25,590 of 30,079. 2DGS-SLAM's corridor output covers every other frame by
  design (2,150 poses).
- Growth videos use an orthographic top view along each reconstruction's own
  estimated display-up axis. Five selected baselines are shown beside each of
  the three long routes; the corridor's two additional baselines remain in the
  asset manifest but are not in the main-page selector.
- Oxford top-view lines are displayed after rigid trajectory-to-released-GT
  alignment, with no scale fitting or new metric claim. FastCaMo-Real has a
  reference surface but no reference trajectory.
- The `RCP1` point format is quantized XYZ + RGB, gzip-compressed and loaded
  on demand. It is a display asset, not an evaluation artifact.
- FastCaMo-Real display clouds sample up to 260,000 source PLY points per
  method and scene. Their viewer point size is 1.5 display pixels; Oxford uses
  1.35 display pixels. These are visual settings, not evaluation settings.

The source exporters are in `scripts/`. Rebuilding the site never modifies
raw datasets, model outputs, benchmark metrics, or pose estimates.
