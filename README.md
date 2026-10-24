<div align="center">

# TKCAM: Text and Keyframe to Camera Trajectory Generation

**NeurIPS 2026** · [Paper (arXiv:2610.11105)](https://arxiv.org/abs/2610.11105) · [Project page](https://linearalgebrayhz.github.io/projects/TKCAM/)

<a href="https://arxiv.org/abs/2610.11105"><img src="https://img.shields.io/badge/arXiv-2610.11105-b31b1b.svg?logo=arxiv" alt="arXiv"></a>
<a href="https://linearalgebrayhz.github.io/projects/TKCAM/"><img src="https://img.shields.io/badge/Project%20Page-TKCAM-blue.svg?logo=googlechrome&logoColor=white" alt="Project Page"></a>
<a href="https://huggingface.co/datasets/yhz-linearalgebra/RealEstate10K-Cap"><img src="https://img.shields.io/badge/Dataset-RealEstate10K--Cap-yellow.svg?logo=huggingface" alt="RealEstate10K-Cap dataset on Hugging Face"></a>
<a href="https://huggingface.co/yhz-linearalgebra/TKCAM"><img src="https://img.shields.io/badge/Checkpoints-TKCAM-yellow.svg?logo=huggingface" alt="TKCAM checkpoints on Hugging Face"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>

</div>

TKCAM generates **camera trajectories** (rotation + translation over time) from text, optionally
conditioned on a first frame or sparse keyframes. It adapts the MoMask residual masked-transformer recipe
from human motion to continuous camera trajectories, and is evaluated with a contrastive
language–trajectory model (CLaTr).

Pipeline: **RVQ-VAE** tokenizes trajectories → **masked transformer** predicts base tokens from text
(+ optional visual conditioning) → **residual transformer** refines the remaining quantizer levels.

## News & release plan
- [x] Training, inference and evaluation code
- [x] Paper on [arXiv](https://arxiv.org/abs/2610.11105)
- [x] [Project page](https://linearalgebrayhz.github.io/projects/TKCAM/)
- [x] [RealEstate10K-Cap dataset](https://huggingface.co/datasets/yhz-linearalgebra/RealEstate10K-Cap)
- [ ] Mixed dataset (RealEstate10K + E.T. + DataDoP) and benchmark split — *coming soon*
- [x] [Pretrained checkpoints](https://huggingface.co/yhz-linearalgebra/TKCAM)

## Installation
Tested on Linux with Python 3.9, PyTorch 2.8 (CUDA 12.8 wheels) and an NVIDIA A100. Requires an NVIDIA
driver compatible with CUDA 12.x. `requirements.txt` pins the PyTorch cu128 extra index and installs
torch 2.8 + CLIP/T5 + all other dependencies, so conda only provides Python (and `ffmpeg`).
```bash
git clone https://github.com/linearalgebrayhz/TKCAM.git
cd TKCAM

# Option A — conda (creates an env named "tkcam")
conda env create -f environment.yml
conda activate tkcam
pip install -e .                          # makes the `tkcam` package importable

# Option B — pip only (any fresh Python 3.9 env)
pip install -r requirements.txt           # torch 2.8 (cu128), CLIP, T5, and all deps
pip install -e .
# MP4 animations need an `ffmpeg` binary on PATH (e.g. `sudo apt install ffmpeg`)
```
Notes:
- The T5 text encoder (`t5-base`) and CLIP weights are downloaded automatically on first use (needs
  internet access, or a pre-populated HuggingFace / CLIP cache).
- The CLaTr evaluator used for metrics is re-implemented in `tkcam/models/evaluator/` and needs no extra
  packages beyond `requirements.txt`.

### Data and checkpoints
The [RealEstate10K-Cap dataset](https://huggingface.co/datasets/yhz-linearalgebra/RealEstate10K-Cap)
and the [pretrained checkpoints](https://huggingface.co/yhz-linearalgebra/TKCAM) are available on
Hugging Face. Download the checkpoints into `checkpoints/` (about 330 MB):
```bash
hf download yhz-linearalgebra/TKCAM --local-dir checkpoints
```
The mixed dataset and benchmark split are **coming soon**. Large data files and checkpoints are not
shipped in this repo (they are gitignored).
All scripts resolve paths **relative to the repository root** by default, so place or symlink your
storage there:
```
TKCAM/
  checkpoints/   -> trained runs: checkpoints/{dataset_name}/{run_name}/  (+ checkpoints/evaluator/ for CLaTr)
  dataset/       -> processed training datasets, e.g. dataset/RealEstate10K_rotmat_25k/
  data/          -> raw sources: data/real-state-10k/, data/ET/et-data/, data/GenDop/DataDop/,
                    data/processed_estate/train_frames/ (RGB frames for visual conditioning)
  Benchmark_hub/ -> benchmark: test.json, reference/, result_<model>/
```
```bash
ln -s /path/to/TKCAM_checkpoints checkpoints
ln -s /path/to/datasets          dataset
ln -s /path/to/raw_data          data
ln -s /path/to/Benchmark_hub     Benchmark_hub
```
Every default can also be overridden on the command line (e.g. `--data_root`, `--frame_dir`,
`--checkpoints-dir`, `--benchmark-root`, `--evaluator-ckpt`). Inference auto-reads the VQ name and
normalization stats from each run's `opt.txt`.

## Repository layout
```
tkcam/        importable package: models/ data/ common/ utils/ options/ motion_loaders/
              + gen_camera.py, compute_dataset_statistics.py (library modules with a CLI)
scripts/      entry points: generate (inference), train_vq, train_t2m_transformer, train_res_transformer,
              train_universal_evaluator, run_tkcam_benchmark_inference,
              run_tkcam_ablation_suite, eval_benchmark_clatr
tools/        dataset construction & conversion (RealEstate10K/ET/GenDoP -> universal 9D, curation, merge)
viz/          trajectory visualization (matplotlib + a Blender renderer)
configs/      example shell launchers for the 3 training stages
legacy/       MoMask-style VQ/transformer eval kept for reference (not part of the pipeline)
docs/         data format & conventions
```
Large artifacts are **not** shipped and are gitignored: `checkpoints/`, `dataset/`, `data/`, `log/`,
`generation/`, `Benchmark_hub/`. Point these at your storage (e.g. symlinks) before training/inference.

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

## Inference
`scripts/generate.py` generates camera trajectories from a text prompt, optionally conditioned on a
first frame or on up to 4 keyframes placed at chosen times:
```bash
# Text only
python scripts/generate.py --mode noframe --duration 6 \
  --text "The camera dollies forward slowly, revealing the living room."

# Text + first frame
python scripts/generate.py --mode firstframe --duration 5 --first_frame room.jpg \
  --text "The camera pans left smoothly across the kitchen."

# Text + keyframes at chosen times (image@seconds, up to 4)
python scripts/generate.py --mode xframe --duration 8 \
  --keyframe hall.jpg@0 --keyframe stairs.jpg@4.5 \
  --text "The camera moves forward through the hallway, then tilts up toward the stairs."
```
- `--duration` is in seconds at 30 fps, rounded to whole 4-frame tokens. Requests below 3 s or above
  11 s are clamped; training clips are ~4.0–9.3 s, so durations outside that range print a warning.
- Each `--mode` loads its own checkpoint pair (`*_25kfixed_{noframe,1frame,xframe}_r4` under
  `checkpoints/realestate10k_rotmat/`, from the [released checkpoints](https://huggingface.co/yhz-linearalgebra/TKCAM));
  override with `--mtrans_name` / `--rtrans_name` (and `--which_epoch` / `--res_which_epoch`, default
  `net_best_fid`). The script checks that the checkpoint matches the mode.
- `--num_samples N` draws N samples (`--seed` for reproducibility); `--no_video` skips the MP4s.
- Outputs per sample in `--out_dir` (default `generation/samples/`): `sample{i}_9d.npy` (`(T, 9)`
  universal layout, see [Data format](#data-format)), `sample{i}_c2w.npy` (`(T, 4, 4)` camera-to-world
  matrices), `sample{i}.mp4`, plus `meta.json` with the prompt, checkpoints, actual duration and the
  keyframe time / frame / token mapping.

### Other entry points
All released models are **T5-conditioned**.
The benchmark script reads the conditioning mode from the checkpoint's `opt.txt`; `gen_camera` needs it
passed explicitly. (These commands are the ones validated in `RELEASE_TEST.md`.)
```bash
# Single prompt -> trajectory .npy + an MP4 animation under generation/<ext>/
python -m tkcam.gen_camera --dataset_name realestate10k_rotmat \
  --name mtrans_25kfixed_noframe_r4 --res_name rtrans_25kfixed_noframe_r4 --conditioning_mode t5 \
  --text_prompt "The camera dollies forward then arcs left around the living room." \
  --gpu_id 0 --ext demo

# Benchmark inference over a prompts JSON (tracks: textonly | firstframe | sparsekeyframe | noframe)
python scripts/run_tkcam_benchmark_inference.py --track noframe \
  --prompts-json <test.json> --out-dir <result_dir> \
  --dataset-name realestate10k_rotmat \
  --mtrans-name mtrans_25kfixed_noframe_r4 --rtrans-name rtrans_25kfixed_noframe_r4 \
  --apply-residual --device cuda:0
# --track firstframe / sparsekeyframe with *_25kfixed_1frame_r4 / *_25kfixed_xframe_r4
```

## Evaluation
```bash
python scripts/eval_benchmark_clatr.py --benchmark-root ./Benchmark_hub \
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
  `opt.txt`; the masked/residual transformers load `net_best_fid.tar` by default.

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
If you use TKCAM, please cite our paper:

```bibtex
@misc{yang2026tkcam,
  title={TKCAM: Text and Keyframe to Camera Trajectory Generation},
  author={Yang, Haozhe and Dou, Zhiyang and Gu, Zekai and Lin, Cheng and Wang, Wenping and Liu, Yuan and Komura, Taku},
  year={2026},
  eprint={2610.11105},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.11105}
}
```

Please also cite the works TKCAM builds on:
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
