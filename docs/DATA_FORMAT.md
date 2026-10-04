# TKCAM data formats & conventions

## Universal 9D (canonical interchange / evaluation)
Per-frame vector `[rot6d(6), x, y, z]`, shape `(T, 9)`:
- `rot6d` = first two **columns** of the 3×3 rotation matrix, laid out `[c0_x,c0_y,c0_z, c1_x,c1_y,c1_z]`.
  Reconstruct R with Gram–Schmidt (`tkcam/utils/unified_data_format.py`, `sixd_to_matrix`); the third
  column = `c0 × c1`.
- `x, y, z` = camera position.
- **Camera-to-world**, **OpenGL** convention (+X right, +Y up, −Z forward), expressed **relative to frame 0**
  (so frame 0 ≈ identity rotation, zero translation).

This is the format for `Benchmark_hub/reference/`, all `result_*` outputs, and CLaTr input.

## 12D rotmat (RVQ training / RealEstate10K-native)
Per-frame `[x, y, z, dx, dy, dz, rot6d(6)]` = position + velocity (Δposition) + rot6d. The RVQ-VAE
trains on this; positions are integrated from velocity at decode. Convert 12D→9D by dropping the velocity
channels and reordering (`tools/convert_rotmat12_to_rotmat9.py`).

## Other representations encountered
- **ET / KITTI** (Exceptional Trajectories): 12 floats per frame = top 3 rows of a 4×4 world-to-camera
  matrix, flattened row-major. `tools/convert_et_to_universal.py`.
- **DataDoP / GenDoP**: shot-based RGBD trajectories; `tools/convert_gendop_to_universal.py`.
- **5D character-centric (CCD baseline)**: `[x, y, z, p_x, p_y]` — camera looks toward a point with a
  yaw/pitch screen offset; cannot express roll. (Lives with the CCD baseline, not TKCAM.)

## Dataset directory layout (a `dataset_name`)
```
{data_root}/
  new_joint_vecs/   per-shot trajectory .npy
  texts/            per-shot captions (.txt); content before the first '#' is the clean caption
  untagged_text/    captions with POS tags stripped (optional)
  metadata/         per-shot metadata (optional)
  Mean.npy Std.npy  normalization statistics
  train.txt val.txt test.txt   split id lists (no trailing newline)
```
Build/convert with `tools/` and merge multiple sources with `tools/merge_universal_datasets.py`
(produces a `merge_manifest.json` recording each id's source).
