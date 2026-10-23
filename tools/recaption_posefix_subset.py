#!/usr/bin/env python3
"""
Re-caption the pose-fixed RealEstate10K dataset with Qwen VL (a pilot subset, or all clips by id range).

Uses the same captioner, prompt and settings as the original 25k run
(QwenVideoCaptioner, PROMPT_TEMPLATE_WITH_SCENE, 32 frames, temperature 0.7);
the only new input is the motion guidance computed from the FIXED trajectory
(guidance.json written by fix_re10k_pose_convention.py).

The subset is stratified by how the guidance changed:
    tilt      tilt words differ between old and fixed guidance
    track     track words differ (and tilt words agree)
    other     guidance differs in another way (e.g. arc <-> pan)
    unchanged guidance identical (control)

Output (inside the posefix dataset dir):
    recaption_subset/subset.json          per-clip record (category, guidances, old/new captions)
    recaption_subset/untagged_text/<id>.txt
    recaption_subset/texts/<id>.txt       "caption#word/POS ..." (same format as tagging.py)

Full dataset, one contiguous slice of the sorted ids per GPU (resumable: clips that
already have a caption are skipped; failed clips are retried one by one, then listed
in recaption_logs/failed_<start>_<end>.json and left without a caption file):
    CUDA_VISIBLE_DEVICES=2 python recaption_posefix_subset.py caption-range \
        --posefix-dir <dir> --start 0 --end 6250
    python recaption_posefix_subset.py tag --posefix-dir <dir> --text-dir <dir>   # writes <dir>/texts

Usage:
    # 1) caption (env with transformers >= 4.57, e.g. tkcam_release_test)
    CUDA_VISIBLE_DEVICES=2 python recaption_posefix_subset.py caption \
        --posefix-dir /data5/haozhe/CamTraj/data/posefix/RealEstate10K_rotmat_25k_posefix
    # 2) POS-tag (env with spaCy + en_core_web_sm, e.g. TKCAM)
    python recaption_posefix_subset.py tag \
        --posefix-dir /data5/haozhe/CamTraj/data/posefix/RealEstate10K_rotmat_25k_posefix
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FRAME_ROOTS = (
    "/data4/haozhe/CamTraj/data/processed_estate/train_frames",
    "/data4/haozhe/CamTraj/data/processed_estate/test_frames",
)
ALWAYS_INCLUDE = ("011805", "000095", "007394", "007482", "000065", "000091")


def words(guidance, kind):
    return {w for w in (f"{kind} left", f"{kind} right", f"{kind} up", f"{kind} down") if w in guidance}


def category(g):
    if g["old"] == g["fixed"]:
        return "unchanged"
    if words(g["old"], "tilts") != words(g["fixed"], "tilts"):
        return "tilt"
    if words(g["old"], "tracks") != words(g["fixed"], "tracks"):
        return "track"
    return "other"


def frame_dir(clip_id, roots=FRAME_ROOTS):
    for root in roots:
        d = Path(root) / clip_id
        if d.is_dir():
            return d
    return None


def select_subset(posefix, per_category, seed):
    guidance = json.loads((posefix / "guidance.json").read_text())
    by_cat = {}
    for sid, g in sorted(guidance.items()):
        by_cat.setdefault(category(g), []).append(sid)
    rng = random.Random(seed)
    chosen = [sid for sid in ALWAYS_INCLUDE if sid in guidance]
    for cat in ("tilt", "track", "other", "unchanged"):
        pool = [s for s in by_cat.get(cat, []) if s not in chosen]
        n = per_category - sum(category(guidance[s]) == cat for s in chosen)
        chosen += rng.sample(pool, max(0, n))
    return chosen, guidance


def cmd_caption(args):
    import torch
    import tkcam.utils.realestate10k_processor as rp

    posefix = Path(args.posefix_dir)
    out = posefix / "recaption_subset"
    (out / "untagged_text").mkdir(parents=True, exist_ok=True)
    mapping = json.loads((posefix / "scene_id_mapping.json").read_text())

    ids, guidance = select_subset(posefix, args.per_category, args.seed)
    torch.manual_seed(args.seed)
    captioner = rp.QwenVideoCaptioner(model_name=args.model, max_frames=args.max_frames, with_scene=True)
    captioner.load_model()

    records = []
    for start in range(0, len(ids), args.batch_size):
        batch = []
        for sid in ids[start:start + args.batch_size]:
            clip_id = mapping[sid]
            meta = json.loads((posefix / "metadata" / f"{sid}.json").read_text())
            fp = captioner.sample_existing_frames(str(frame_dir(clip_id)))
            batch.append((fp, guidance[sid]["fixed"], sid, meta["num_frames"]))
        results = captioner.generate_captions_batch(batch)
        for (_, _, sid, _), res in zip(batch, results):
            assert res["scene_id"] == sid
            caption = res["caption"]
            (out / "untagged_text" / f"{sid}.txt").write_text(caption)
            records.append({
                "id": sid,
                "re10k_clip_id": mapping[sid],
                "category": category(guidance[sid]),
                "guidance_old": guidance[sid]["old"],
                "guidance_fixed": guidance[sid]["fixed"],
                "caption_old": (posefix / "untagged_text_old" / f"{sid}.txt").read_text().strip(),
                "caption_new": caption,
                "success": bool(res.get("success")),
            })
        captioner.cleanup_temp_images()
        print(f"captioned {len(records)}/{len(ids)}", flush=True)

    (out / "subset.json").write_text(json.dumps(
        {"model": args.model, "max_frames": args.max_frames, "seed": args.seed,
         "generate_kwargs": rp.QwenVideoCaptioner.GENERATE_KWARGS, "clips": records}, indent=1))


def cmd_caption_range(args):
    import time
    import torch
    import tkcam.utils.realestate10k_processor as rp

    posefix = Path(args.posefix_dir)
    out = posefix / "untagged_text"
    out.mkdir(exist_ok=True)
    logs = posefix / "recaption_logs"
    logs.mkdir(exist_ok=True)
    mapping = json.loads((posefix / "scene_id_mapping.json").read_text())
    guidance = json.loads((posefix / "guidance.json").read_text())

    ids = sorted(guidance)[args.start:args.end]
    if args.ids:
        ids = [s for s in ids if s in set(args.ids)]
    roots = (args.frame_root,) if args.frame_root else FRAME_ROOTS
    todo = [s for s in ids if not ((out / f"{s}.txt").exists() and (out / f"{s}.txt").stat().st_size > 0)]
    print(f"range [{args.start}, {args.end}): {len(ids)} clips, {len(todo)} to caption", flush=True)

    torch.manual_seed(args.seed + args.start)
    captioner = rp.QwenVideoCaptioner(model_name=args.model, max_frames=args.max_frames, with_scene=True)
    captioner.load_model()

    def item(sid):
        meta = json.loads((posefix / "metadata" / f"{sid}.json").read_text())
        fp = captioner.sample_existing_frames(str(frame_dir(mapping[sid], roots)))
        return (fp, guidance[sid]["fixed"], sid, meta["num_frames"])

    failed = {}
    t0 = time.time()
    done = 0
    for start in range(0, len(todo), args.batch_size):
        batch = [item(sid) for sid in todo[start:start + args.batch_size]]
        results = captioner.generate_captions_batch(batch)
        retry = []
        for b, res in zip(batch, results):
            if res.get("success") and res["scene_id"] == b[2] and res["caption"]:
                (out / f"{b[2]}.txt").write_text(res["caption"])
            else:
                retry.append(b)
        for b in retry:     # one by one, e.g. after an OOM in the batched call
            res = captioner.generate_caption_with_context(*b)
            if res.get("success") and res["caption"]:
                (out / f"{b[2]}.txt").write_text(res["caption"])
            else:
                failed[b[2]] = res.get("error", "unknown")
        captioner.cleanup_temp_images()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        done += len(batch)
        rate = (time.time() - t0) / done
        print(f"[{args.start}:{args.end}] {done}/{len(todo)}  {rate:.2f}s/clip  "
              f"eta {rate * (len(todo) - done) / 3600:.2f}h  failed {len(failed)}  "
              f"peak_mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
    (logs / f"failed_{args.start}_{args.end}.json").write_text(json.dumps(failed, indent=1))
    print("finished", flush=True)


def cmd_tag(args):
    import spacy

    nlp = spacy.load("en_core_web_sm")
    root = Path(args.text_dir) if args.text_dir else Path(args.posefix_dir) / "recaption_subset"
    (root / "texts").mkdir(exist_ok=True)
    paths = sorted((root / "untagged_text").glob("*.txt"))
    texts = [p.read_text(encoding="utf-8") for p in paths]
    for path, text, doc in zip(paths, texts, nlp.pipe(texts, batch_size=256)):
        tags = [f"{t.text}/{t.pos_}" for t in doc if not t.is_punct]
        (root / "texts" / path.name).write_text(f"{text.strip()}#{' '.join(tags)}\n", encoding="utf-8")
    print("tagged", len(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["caption", "caption-range", "tag"])
    parser.add_argument("--posefix-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--per-category", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0, help="caption-range: first index into the sorted ids")
    parser.add_argument("--end", type=int, default=None, help="caption-range: end index (exclusive)")
    parser.add_argument("--ids", nargs="*", help="caption-range: only these dataset ids")
    parser.add_argument("--frame-root", default=None,
                        help="caption-range: folder with one frame folder per RE10K clip (default: train/test_frames)")
    parser.add_argument("--text-dir", default=None, help="tag: folder holding untagged_text/ (default: the pilot subset)")
    args = parser.parse_args()
    {"caption": cmd_caption, "caption-range": cmd_caption_range, "tag": cmd_tag}[args.command](args)


if __name__ == "__main__":
    main()
