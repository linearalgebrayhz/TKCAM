# TKCAM release — clean-room first-time test

Goal: simulate a new user cloning TKCAM and getting **inference** running, in a **fresh conda env**
(separate from the dev env), to validate the install instructions and code paths.

Date: 2026-06-20. Machine: A100 (driver 555.42 / CUDA 12.5).

## Procedure (exactly what a first-timer runs)
```bash
conda env create -n tkcam_release_test -f environment.yml   # Python 3.9 + pip -> requirements.txt
conda activate tkcam_release_test
pip install -e .
ln -s /path/to/TKCAM_checkpoints checkpoints                # obtain pretrained checkpoints

# 1) benchmark inference (text-only / "noframe" headline model)
python scripts/run_tkcam_benchmark_inference.py --track noframe \
  --dataset-name realestate10k_rotmat \
  --mtrans-name mtrans_25k_noframe_r4 --rtrans-name rtrans_25k_noframe_r4 \
  --prompts-json /path/to/test.json --out-dir out --apply-residual --max-samples 5 --device cuda:0

# 2) single-prompt generation (+ auto MP4 animation)
python -m tkcam.gen_camera --dataset_name realestate10k_rotmat \
  --name mtrans_25k_noframe_r4 --res_name rtrans_25k_noframe_r4 --conditioning_mode t5 \
  --text_prompt "The camera dollies forward then arcs left around the living room." \
  --gpu_id 0 --ext demo
```

## Result: PASS
- env built from `environment.yml`; `torch 2.8.0+cu128` (CUDA available), `transformers 4.57.1`, T5 import OK.
- benchmark inference → 5 valid 9D `.npy` (e.g. `000157.npy` shape (204, 9), finite).
- single-prompt generation → trajectory `.npy` (200, 12 raw) + an MP4 animation
  (`generation/demo/animations/0/sample0_repeat0_len200_pred_velint.mp4`, ~1.2 MB).

## Issues found and fixed (would have blocked a first-timer)
1. **`environment.yml`** was a non-portable full `conda env export` (221 build-hash pins) with a
   self-conflicting `mkl`<2023 vs `llvm-openmp` → conda fails to solve on a clean machine.
   → Replaced with a minimal portable env (Python + pip → `requirements.txt`).
2. **`requirements.txt` pinned the wrong stack** (`torch==1.12.0` / cu113). The released results were
   produced with **torch 2.8 / cu128**, and `transformers` ≥4.5x needs torch ≥2.x for the T5
   text-conditioning backend (with torch 1.12 it raises "T5EncoderModel requires PyTorch").
   → Repinned to torch 2.8 (cu128) + matching transformers/tokenizers/hf-hub/safetensors/sentencepiece.
3. **`transformers` + `sentencepiece` were missing** from `requirements.txt` (T5 conditioning).  → Added.
4. **Checkpoints default path** resolved to `scripts/checkpoints` after the repo reorg (the README
   documents symlinking `checkpoints` at the repo root). → Fixed `run_tkcam_benchmark_inference.py`
   `_resolve_checkpoints_dir` to use the repo root.

## Notes for users
- The released "noframe" model is **T5-conditioned**; for `gen_camera` pass `--conditioning_mode t5`
  (the benchmark script auto-reads the mode from the checkpoint `opt.txt`, so `--track noframe` needs no
  flag). t5-base is downloaded from HuggingFace on first run (needs internet).
- CUDA 12.x driver required (cu128 wheels). The `checkpoints/` symlink is gitignored (not shipped).
