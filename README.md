# TKCAM — Masked Generative Modeling of Camera Trajectories

TKCAM generates **camera trajectories** (rotation + translation over time) from text, optionally
conditioned on a first frame or sparse keyframes. It adapts the MoMask residual masked-transformer recipe
from human motion to continuous camera trajectories, and is evaluated with a contrastive
language–trajectory model (CLaTr).

Pipeline: **RVQ-VAE** tokenizes trajectories → **masked transformer** predicts base tokens from text
(+ optional visual conditioning) → **residual transformer** refines the remaining quantizer levels.

## Install
Requires an NVIDIA driver compatible with CUDA 12.x wheels. `requirements.txt` pins the PyTorch
cu128 extra index and installs torch 2.8 + CLIP/T5 + all deps, so conda only provides Python.
(Transformers ≥4.5x needs torch ≥2.x for the T5 text-conditioning backend.)
```bash
# Option A — conda (creates an env named "tkcam")
conda env create -f environment.yml
conda activate tkcam
pip install -e .                          # makes the `tkcam` package importable

# Option B — pip only (any fresh Python 3.9 env)
pip install -r requirements.txt           # torch 1.12 (cu113), CLIP, and all deps
pip install -e .

# Optional: CLaTr evaluator extras (only for eval_benchmark_clatr / training metrics)
pip install -r requirements_clatr.txt     # note: pytorch-lightning==1.7.7 (not 2.x)
```

### Pretrained checkpoints
Checkpoints/data are not shipped (gitignored). Place (or symlink) trained runs under `checkpoints/`
following `checkpoints/{dataset_name}/{run_name}/`; inference auto-reads the VQ name and normalization
stats from each run's `opt.txt`. Example:
```bash
ln -s /path/to/TKCAM_checkpoints checkpoints
```

## Repository layout
```
tkcam/        importable package: models/ data/ common/ utils/ options/ motion_loaders/
              + gen_camera.py, compute_dataset_statistics.py (library modules with a CLI)
scripts/      entry points: train_vq, train_t2m_transformer, train_res_transformer,
              train_universal_evaluator, run_tkcam_benchmark_inference,
              run_tkcam_ablation_suite, eval_benchmark_clatr
tools/        dataset construction & conversion (RealEstate10K/ET/GenDoP -> universal 9D, curation, merge)
viz/          trajectory visualization (matplotlib + a Blender renderer)
configs/      example shell launchers for the 3 training stages
legacy/       MoMask-style VQ/transformer eval kept for reference (not part of the pipeline)
docs/         data format & conventions
```
Large artifacts are **not** shipped and are gitignored: `checkpoints/`, `dataset/`, `log/`,
`generation/`. Point these at your storage (e.g. symlinks) before training/inference.

## Data format
Trajectories use the **universal 9D** layout `[rot6d(6), x, y, z]` per frame (camera-to-world, OpenGL,
first-frame-relative); `rot6d` = first two columns of the rotation matrix (Gram-Schmidt to reconstruct).
The RVQ trains on a 12D rotmat variant `[x,y,z, dx,dy,dz, rot6d]`. See `docs/DATA_FORMAT.md` and
`tkcam/utils/unified_data_format.py`. Convert source datasets with `tools/convert_*` and merge with
`tools/merge_universal_datasets.py`.

## Training (3 stages)
```bash
# Stage 1 — RVQ-VAE tokenizer
python scripts/train_vq.py --dataset_name realestate10k_rotmat --name rvq_tkcam \
  --num_quantizers 6 --batch_size 512 --gpu_id 0 --evaluator_ckpt <clatr.ckpt>

# Stage 2 — masked transformer (text-only shown; add --use_first_frame or
#           --use_sparse_frames --max_sparse_frames N for visual conditioning)
python scripts/train_t2m_transformer.py --dataset_name realestate10k_rotmat --name mtrans_tkcam \
  --vq_name rvq_tkcam --conditioning_mode clip --gpu_id 0 --evaluator_ckpt <clatr.ckpt>

# Stage 3 — residual transformer
python scripts/train_res_transformer.py --dataset_name realestate10k_rotmat --name rtrans_tkcam \
  --vq_name rvq_tkcam --share_weight --gpu_id 0 --evaluator_ckpt <clatr.ckpt>
```
The CLaTr evaluator used for FID/retrieval metrics during training is trained with
`scripts/train_universal_evaluator.py`. Example launchers are in `configs/`.

## Generation
The released text-only model (`mtrans_25k_noframe_r4` / `rtrans_25k_noframe_r4`) is **T5-conditioned**.
The benchmark script reads the conditioning mode from the checkpoint's `opt.txt`; `gen_camera` needs it
passed explicitly. (These commands are the ones validated in `RELEASE_TEST.md`.)
```bash
# Single prompt -> trajectory .npy + an MP4 animation under generation/<ext>/
python -m tkcam.gen_camera --dataset_name realestate10k_rotmat \
  --name mtrans_25k_noframe_r4 --res_name rtrans_25k_noframe_r4 --conditioning_mode t5 \
  --text_prompt "The camera dollies forward then arcs left around the living room." \
  --gpu_id 0 --ext demo

# Benchmark inference over a prompts JSON (tracks: textonly | firstframe | sparsekeyframe | noframe)
python scripts/run_tkcam_benchmark_inference.py --track noframe \
  --prompts-json <test.json> --out-dir <result_dir> \
  --dataset-name realestate10k_rotmat \
  --mtrans-name mtrans_25k_noframe_r4 --rtrans-name rtrans_25k_noframe_r4 \
  --apply-residual --device cuda:0
```

## Evaluation
```bash
python scripts/eval_benchmark_clatr.py --benchmark-root <Benchmark_hub> \
  --evaluator-ckpt <clatr.ckpt> --device cuda:0
# add --dump-per-sample <dir> to also write per-sample matching scores + retrieval ranks (CSV)
```

## Conditioning modes
- `--conditioning_mode {clip,t5,id_embedding}` — text encoder for captions.
- `--use_first_frame` — condition on the first RGB frame (CLIP image features).
- `--use_sparse_frames --max_sparse_frames N` — condition on up to N keyframes.

## Notes
- `viz/visualize_clatr9d_blender.py` runs **inside Blender** (`blender --background --python ...`); it
  imports `bpy` and will not run under plain Python.
- Checkpoints follow `checkpoints/{dataset_name}/{run_name}/model/` with `meta/Mean.npy`/`Std.npy` and
  `opt.txt`; the masked/residual transformers select `net_best_fid.tar`.

## Acknowledgements
TKCAM builds directly on excellent prior work — please cite/credit them:
- **MoMask** (Guo et al., CVPR 2024) — our code base; the residual VQ-VAE + masked/residual transformer
  recipe is adapted from MoMask to camera trajectories.
  https://github.com/EricGuo5513/momask-codes
- **CLaTr** and **E.T. (The Exceptional Trajectories)** (Courant et al.) — we use CLaTr as the
  contrastive language–trajectory evaluator for our metrics, and the E.T. trajectories are part of our
  mixed benchmark (the DIRECTOR baseline is also from this line of work).
  - CLaTr: https://github.com/robin-courant/CLaTr
  - E.T. dataset/project: https://github.com/robin-courant/the-exceptional-trajectories
  - DIRECTOR: https://github.com/robin-courant/DIRECTOR

## Citation
```bibtex
@inproceedings{guo2024momask,
  title={MoMask: Generative Masked Modeling of 3D Human Motions},
  author={Guo, Chuan and Mu, Yuxuan and Javed, Muhammad Gohar and Wang, Sen and Cheng, Li},
  booktitle={CVPR}, year={2024}
}
@inproceedings{courant2024et,
  title={E.T. the Exceptional Trajectories: Text-to-camera-trajectory generation with character awareness},
  author={Courant, Robin and Dufour, Nicolas and Wang, Xi and Christie, Marc and Kalogeiton, Vicky},
  booktitle={ECCV}, year={2024}
}
```
