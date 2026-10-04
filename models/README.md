# Model Weights

Model weights are not stored in this repository (GitHub file size limits). Download them manually and place into the directories listed below. SHA-256 checksums are provided for verification.

| File | Size | Target path | SHA-256 |
|---|---|---|---|
| `pose_landmarker_full.task` | 9.4 MB | `models/` | `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad` |
| `yolox_m_humanart.onnx` | 99 MB | `models/` | `3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23` |
| `rtmpose_m_body7_256x192.onnx` | 53 MB | `models/` | `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c` |
| `rtmo_m_body7_640x640.onnx` | 87 MB | `models/` | `750d872151ff4652f02c738efc3a547e112ce9b688fe920bc17f948e8c3afdac` |
| `pose_iter_584000.caffemodel` | 100 MB | `openpose/models/pose/body_25/` | `44e3d7ebd8c8b62d4366d67127f1b562611a9e8fd0f4f3cdeeb4bb4a6ed12be6` |

## Download links

**BlazePose** (MediaPipe Tasks API, `run_blazepose.py`):
```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

**RTMPose** (rtmlib, `run_rtmpose.py`) — detector + pose model (unpack the zip, take the `.onnx` inside):
```
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip
```

**RTMO** (rtmlib, `run_rtmo.py`):
```
https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip
```

**OpenPose** (`run_openpose.py`):
- Portable build 1.7.0 win64 CPU: unpack into the project root as `openpose/`
  ```
  https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases
  ```
- BODY_25 weights — the official CMU server (`posefs1.perception.cs.cmu.edu`) is currently unreachable; use the Hugging Face mirror:
  ```
  https://huggingface.co/camenduru/openpose/resolve/main/models/pose/body_25/pose_iter_584000.caffemodel
  ```

## Notes

- The OpenPose `caffemodel` SHA-256 matches the checksum in the official OpenPose LFS metadata.
- On Windows, if `curl` fails against `huggingface.co` with a certificate revocation error, add `--ssl-revoke-best-effort`.
