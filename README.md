# hpe-motion-comparison

Comparative study of four human pose estimation (HPE) models — **OpenPose**, **BlazePose**, **RTMPose**, **RTMO** — evaluated as feature extractors for a temporal motion representation `v_t` (per-frame movement intensity of head, left/right arm, left/right leg), intended for audio-video synchronization tasks.

All models run on **CPU** with a unified protocol: same video, canonical COCO-17 keypoint set, same normalization (mid-hip anchor, torso-length scale), confidence threshold 0.3, no interpolation of missing frames.

## Repository layout

| Path | Purpose |
|---|---|
| `common.py` | Shared constants: COCO-17 keypoints, functional groups, native→COCO-17 mappings, CSV schema |
| `run_blazepose.py` | BlazePose via MediaPipe Tasks API (`PoseLandmarker`, VIDEO mode) |
| `run_rtmpose.py` | RTMPose via rtmlib (top-down: YOLOX detector + pose model) |
| `run_rtmo.py` | RTMO via rtmlib (one-stage detector+pose) |
| `run_openpose.py` | OpenPose 1.7.0 portable build (bottom-up, PAF), external `OpenPoseDemo.exe` |
| `analyze_motion.py` | Model-independent feature extraction: keypoints → normalized `v_t` series |
| `cross_model_analysis.py` | Cross-model comparison: detection rate, m_j statistics, D_j, autocorrelation, correlations on the common frame subset |
| `point_stats.py` | Per-keypoint statistics: confidence distribution, out-of-frame points, left/right asymmetry |
| `rtmpose_diagnostics.py` | Outlier diagnostics for RTMPose: localization errors vs normalization-scale jumps |
| `cross_timeseries_plot.py` | Cross-model time-series plots (`m_head`, torso scale `s_t`) |
| `check_blazepose_gpu.py` | Utility: check GPU delegate availability for MediaPipe (not used in the experiment) |
| `models/README.md` | Download links and SHA-256 for all model weights |
| `results/` | Precomputed outputs of all runs (see below) |
| `data/` | Source video — **not included** (standard YouTube license); put your own video here |

## Reproducing the experiment

### 1. Environment

Python 3.14, virtual environment:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Weights and OpenPose build

Download model weights per `models/README.md` (links + SHA-256 checksums), unpack the OpenPose portable build into the project root as `openpose/`.

### 3. Video

Place the test video as `data/test_dance_30s.mp4` (1920×1080, 25 fps, ~30 s, single person, static camera). The original video is not redistributed here for licensing reasons; any video matching the protocol above can be substituted.

### 4. Run the models

Each runner writes `results/<model>/keypoints.csv` (frame, time_s, then x/y/c for 17 COCO keypoints in pixels, NaN for misses) and `timing.json` (device, model, wall-clock time of the measured pass):

```
python run_blazepose.py --preview
python run_rtmo.py --preview
python run_rtmpose.py --preview
python run_openpose.py --preview
```

Notes:

- `--preview` renders an annotated `preview.mp4` (skeleton only, for checking); it is optional.
- OpenPose is an external process: each invocation reloads the model, so it uses a shortened warm-up instead of the double full pass used by the other models (recorded in `timing.json`).
- Approximate CPU runtimes for the 751-frame clip (i9-13980HX): BlazePose ~21 s (35 FPS), RTMO ~60 s (12.6 FPS), RTMPose ~530 s (1.4 FPS), OpenPose ~3.8 h (0.05 FPS).

### 5. Extract motion features

For each model:

```
python analyze_motion.py results/blazepose
python analyze_motion.py results/rtmo
python analyze_motion.py results/rtmpose
python analyze_motion.py results/openpose
```

Produces `motion_features.csv` (per-frame `v_t`), `summary.json` (mean/median `m_j`, `D_j`, validity) and `motion_timeseries.png` in the model folder.

### 6. Cross-model analysis

```
python cross_model_analysis.py
python point_stats.py
python cross_timeseries_plot.py
python rtmpose_diagnostics.py
```

Writes to `results/`:

| Output | Content |
|---|---|
| `cross_model_report.txt`, `cross_model_summary.json` | Full-series vs common-subset (`V_common`) statistics, D_j, lag-1 autocorrelation, pairwise Pearson correlations (raw and smoothed, window 5) |
| `point_stats_report.txt` | Per-keypoint confidence and out-of-frame statistics |
| `cross_motion_timeseries.png`, `cross_scale_timeseries.png` | Time series of `m_head` and torso scale `s_t` across the four models |
| `rtmpose_spike_report.txt`, `rtmpose/rtmpose_head_decomposition.png` | Decomposition of RTMPose outliers into localization errors vs scale jumps |

## Results included in this repository

`results/` contains the complete outputs of the reported experiment: `keypoints.csv`, `motion_features.csv`, `summary.json`, `timing.json` and plots for each model, plus all cross-model reports. Not included: source video, rendered previews (license), raw OpenPose per-frame JSONs (redundant with `keypoints.csv`), and model weights.
