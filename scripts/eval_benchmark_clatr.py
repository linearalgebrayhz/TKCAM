#!/usr/bin/env python3
"""
Evaluate benchmark baseline trajectory folders with a trained CLaTr evaluator.

Reads:
1) Benchmark_hub/test.json for text prompts + ids
2) Benchmark_hub/result_<baseline>/ for predicted trajectories (*.npy)

Per baseline, computes retrieval-style metrics between text and trajectory
embeddings from the evaluator checkpoint, and writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tkcam.models.evaluator.clatr_models import CLaTrEvalWrapper, compute_diversity, compute_fid


_ROOT = Path(__file__).resolve().parent


@dataclass
class EvalSample:
    sid: str
    text: str
    traj: np.ndarray  # (T, D)


def _resolve_benchmark_root(cli_path: Optional[str]) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    return (_ROOT.parent / "Benchmark_hub").resolve()


def _find_prompts_file(benchmark_root: Path, cli_prompts_path: Optional[str]) -> Path:
    if cli_prompts_path:
        p = Path(cli_prompts_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Prompts JSON not found: {p}")
        return p
    p = benchmark_root / "test.json"
    if not p.is_file():
        raise FileNotFoundError(f"Prompts JSON not found: {p}")
    return p


def _load_prompts(prompts_path: Path) -> List[Dict[str, Any]]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise ValueError(
        f"Unsupported prompt json format at {prompts_path}; expected list or {{'items': [...]}}"
    )


def _resolve_ckpt_file(path_or_dir: str) -> Path:
    p = Path(path_or_dir).expanduser().resolve()
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"Evaluator checkpoint path does not exist: {p}")

    # Standard output of train_evaluator.py
    for name in ("best.pt", "latest.pt"):
        cand = p / name
        if cand.is_file():
            return cand

    epoch_ckpts = sorted(p.glob("epoch_*.pt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]

    raise FileNotFoundError(
        f"No evaluator checkpoint found under directory: {p}. "
        "Expected one of best.pt/latest.pt/epoch_*.pt"
    )


def _discover_baseline_dirs(
    benchmark_root: Path,
    baseline_names: Optional[Sequence[str]],
    result_prefix: str,
) -> List[Tuple[str, Path]]:
    if baseline_names:
        out = []
        for name in baseline_names:
            d = benchmark_root / f"{result_prefix}{name}"
            if not d.is_dir():
                raise FileNotFoundError(f"Baseline directory not found: {d}")
            out.append((name, d))
        return out

    # Auto discover all result_* folders.
    discovered = []
    for d in sorted(benchmark_root.iterdir()):
        if d.is_dir() and d.name.startswith(result_prefix):
            bname = d.name[len(result_prefix) :]
            discovered.append((bname, d))
    if not discovered:
        raise FileNotFoundError(
            f"No baseline folders found in {benchmark_root} with prefix '{result_prefix}'"
        )
    return discovered


def _parse_baseline_names(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    names = [x.strip() for x in raw.split(",") if x.strip()]
    return names or None


def _load_baseline_samples(
    prompts: Sequence[Dict[str, Any]],
    result_dir: Path,
    max_samples: Optional[int],
    strict_9d: bool,
    log: logging.Logger,
) -> Tuple[List[EvalSample], Dict[str, Any]]:
    if max_samples is not None:
        prompts = prompts[: int(max_samples)]

    loaded: List[EvalSample] = []
    missing_ids: List[str] = []
    bad_ids: List[str] = []

    for i, item in enumerate(prompts):
        sid = str(item.get("id", item.get("ID", f"row_{i}"))).strip()
        text = str(item.get("text", item.get("prompt", ""))).strip()
        if not sid:
            sid = f"row_{i}"
        if not text:
            continue

        npy_path = result_dir / f"{sid}.npy"
        if not npy_path.is_file():
            missing_ids.append(sid)
            continue

        try:
            arr = np.load(npy_path)
            if arr.ndim != 2:
                raise ValueError(f"expected 2D array, got shape {arr.shape}")
            if strict_9d and arr.shape[1] != 9:
                raise ValueError(f"strict_9d enabled, expected D=9, got D={arr.shape[1]}")
            if arr.shape[1] not in (9, 12):
                raise ValueError(f"expected D in (9,12), got D={arr.shape[1]}")
            if arr.shape[0] < 2:
                raise ValueError(f"trajectory too short T={arr.shape[0]}")
            loaded.append(EvalSample(sid=sid, text=text, traj=arr.astype(np.float32)))
        except Exception as exc:
            bad_ids.append(sid)
            log.warning("Skip invalid npy for id=%s at %s: %r", sid, npy_path, exc)

    stats = {
        "requested_items": len(prompts),
        "loaded_items": len(loaded),
        "missing_items": len(missing_ids),
        "bad_items": len(bad_ids),
        "missing_ids_sample": missing_ids[:20],
        "bad_ids_sample": bad_ids[:20],
    }
    return loaded, stats


def _pad_trajectories(samples: Sequence[EvalSample]) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    lengths = [int(s.traj.shape[0]) for s in samples]
    max_t = max(lengths)
    feat_dim = int(samples[0].traj.shape[1])
    batch = np.zeros((len(samples), max_t, feat_dim), dtype=np.float32)

    for i, s in enumerate(samples):
        t = s.traj.shape[0]
        batch[i, :t] = s.traj

    motions = torch.from_numpy(batch)
    m_lens = torch.tensor(lengths, dtype=torch.long)
    ids = [s.sid for s in samples]
    texts = [s.text for s in samples]
    return motions, m_lens, ids, texts


def _encode_text_in_chunks(
    wrapper: CLaTrEvalWrapper,
    texts: Sequence[str],
    chunk_size: int,
) -> np.ndarray:
    out = []
    for i in range(0, len(texts), chunk_size):
        chunk = list(texts[i : i + chunk_size])
        out.append(wrapper.get_text_embeddings(chunk))
    return np.concatenate(out, axis=0)


def _encode_traj_in_chunks(
    wrapper: CLaTrEvalWrapper,
    motions: torch.Tensor,
    m_lens: torch.Tensor,
    chunk_size: int,
) -> np.ndarray:
    out = []
    n = motions.shape[0]
    for i in range(0, n, chunk_size):
        out.append(wrapper.get_trajectory_embeddings(motions[i : i + chunk_size], m_lens[i : i + chunk_size]))
    return np.concatenate(out, axis=0)


def _recall_and_rank_metrics(sim: np.ndarray, return_ranks: bool = False):
    """
    sim: (N_text, N_traj), paired ground-truth is diagonal i<->i

    If return_ranks=True, returns (metrics, t2m_ranks, m2t_ranks) with per-sample 1-indexed ranks.
    """
    n = sim.shape[0]
    if n == 0 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"Expected square non-empty sim matrix, got {sim.shape}")

    # text -> traj
    t2m_ranks = []
    for i in range(n):
        order = np.argsort(-sim[i])  # descending similarity
        rank = int(np.where(order == i)[0][0]) + 1  # 1-indexed rank
        t2m_ranks.append(rank)
    t2m_ranks = np.asarray(t2m_ranks, dtype=np.int64)

    # traj -> text
    m2t_ranks = []
    sim_t = sim.T
    for i in range(n):
        order = np.argsort(-sim_t[i])
        rank = int(np.where(order == i)[0][0]) + 1
        m2t_ranks.append(rank)
    m2t_ranks = np.asarray(m2t_ranks, dtype=np.int64)

    def _r_at_k(ranks: np.ndarray, k: int) -> float:
        return float(100.0 * np.mean(ranks <= k))

    metrics = {
        "t2m/R01": _r_at_k(t2m_ranks, 1),
        "t2m/R02": _r_at_k(t2m_ranks, 2),
        "t2m/R03": _r_at_k(t2m_ranks, 3),
        "t2m/R05": _r_at_k(t2m_ranks, 5),
        "t2m/R10": _r_at_k(t2m_ranks, 10),
        "t2m/MedR": float(np.median(t2m_ranks)),
        "t2m/MeanR": float(np.mean(t2m_ranks)),
        "m2t/R01": _r_at_k(m2t_ranks, 1),
        "m2t/R02": _r_at_k(m2t_ranks, 2),
        "m2t/R03": _r_at_k(m2t_ranks, 3),
        "m2t/R05": _r_at_k(m2t_ranks, 5),
        "m2t/R10": _r_at_k(m2t_ranks, 10),
        "m2t/MedR": float(np.median(m2t_ranks)),
        "m2t/MeanR": float(np.mean(m2t_ranks)),
    }
    if return_ranks:
        return metrics, t2m_ranks, m2t_ranks
    return metrics


def _evaluate_one_baseline(
    *,
    baseline_name: str,
    samples: Sequence[EvalSample],
    wrapper: CLaTrEvalWrapper,
    embed_batch_size: int,
) -> Dict[str, Any]:
    motions, m_lens, ids, texts = _pad_trajectories(samples)

    text_emb = _encode_text_in_chunks(wrapper, texts=texts, chunk_size=embed_batch_size)
    traj_emb = _encode_traj_in_chunks(
        wrapper,
        motions=motions,
        m_lens=m_lens,
        chunk_size=embed_batch_size,
    )

    sim = text_emb @ traj_emb.T  # embeddings are already L2-normalized in wrapper
    retrieval, t2m_ranks, m2t_ranks = _recall_and_rank_metrics(sim, return_ranks=True)
    diag = np.diag(sim)

    metrics = {
        **retrieval,
        "matching_score": float(np.mean(diag)),
        "traj_diversity": float(compute_diversity(traj_emb, seed=0)),
        "num_evaluated": int(len(samples)),
        "avg_traj_len": float(np.mean(m_lens.numpy())),
        "min_traj_len": int(np.min(m_lens.numpy())),
        "max_traj_len": int(np.max(m_lens.numpy())),
    }
    return {
        "baseline": baseline_name,
        "metrics": metrics,
        "sample_ids": ids,
        "traj_embeddings": traj_emb,
        "text_embeddings": text_emb,
        "per_sample": {
            "id": list(ids),
            "matching_score": diag.astype(np.float32),
            "t2m_rank": t2m_ranks,
            "m2t_rank": m2t_ranks,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate result_<baseline> benchmark directories with a trained CLaTr evaluator checkpoint."
    )
    parser.add_argument("--benchmark-root", type=str, default=None, help="Path to Benchmark_hub.")
    parser.add_argument("--prompts-json", type=str, default=None, help="Path to test.json.")
    parser.add_argument(
        "--evaluator-ckpt",
        type=str,
        default="/home/haozhe/CamTraj/momask-codes/checkpoints/evaluator/clatr_universal_eval_RE10K_ET_GenDoP_9d_ld256_od512",
        help="Path to evaluator checkpoint file (.pt) or directory containing best.pt/latest.pt.",
    )
    parser.add_argument(
        "--baseline-names",
        type=str,
        default=None,
        help="Comma-separated names, e.g. 'CCD,director,GenDoP_noRGB'. If omitted, auto-discovers result_* folders.",
    )
    parser.add_argument("--result-prefix", type=str, default="result_", help="Baseline folder prefix.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only evaluate first N prompts.")
    parser.add_argument("--embed-batch-size", type=int, default=128, help="Embedding chunk size.")
    parser.add_argument(
        "--min-eval-items",
        type=int,
        default=32,
        help="Require at least this many valid matched items to evaluate a baseline.",
    )
    parser.add_argument("--strict-9d", action="store_true", help="Fail samples that are not D=9.")
    parser.add_argument(
        "--reference-result-dir",
        type=str,
        default=None,
        help=(
            "Optional reference trajectory folder for FID comparison "
            "(expects <id>.npy, typically GT trajectories)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Where to write metrics JSON. Default: <benchmark_root>/benchmark_clatr_eval.json",
    )
    parser.add_argument(
        "--dump-per-sample",
        type=str,
        default=None,
        help=(
            "Optional directory; if set, writes <dir>/<baseline>_per_sample.csv with per-sample "
            "CLaTr matching score and t2m/m2t ranks (for human-preference correlation, Study B)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("eval_benchmark_clatr")

    benchmark_root = _resolve_benchmark_root(args.benchmark_root)
    prompts_path = _find_prompts_file(benchmark_root, args.prompts_json)
    prompts = _load_prompts(prompts_path)
    if args.max_samples is not None:
        prompts = prompts[: int(args.max_samples)]
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    ckpt_file = _resolve_ckpt_file(args.evaluator_ckpt)
    device = torch.device(args.device)
    log.info("Loading evaluator from %s on %s", ckpt_file, device)
    wrapper = CLaTrEvalWrapper(ckpt_path=str(ckpt_file), device=device)

    baseline_names = _parse_baseline_names(args.baseline_names)
    baselines = _discover_baseline_dirs(
        benchmark_root=benchmark_root,
        baseline_names=baseline_names,
        result_prefix=str(args.result_prefix),
    )
    log.info("Will evaluate %d baseline directories", len(baselines))

    report: Dict[str, Any] = {
        "prompts_json": str(prompts_path),
        "num_prompts_considered": len(prompts),
        "evaluator_ckpt": str(ckpt_file),
        "device": str(device),
        "baselines": {},
    }

    reference_embeddings: Optional[Dict[str, np.ndarray]] = None
    if args.reference_result_dir:
        ref_dir = Path(args.reference_result_dir).expanduser().resolve()
        if not ref_dir.is_dir():
            raise FileNotFoundError(f"reference-result-dir not found: {ref_dir}")
        log.info("Loading reference trajectories from %s", ref_dir)
        ref_samples, ref_stats = _load_baseline_samples(
            prompts=prompts,
            result_dir=ref_dir,
            max_samples=None,
            strict_9d=bool(args.strict_9d),
            log=log,
        )
        if len(ref_samples) < int(args.min_eval_items):
            raise ValueError(
                f"Reference dir has too few valid samples: {len(ref_samples)} "
                f"(min-eval-items={int(args.min_eval_items)})"
            )
        ref_motions, ref_lens, ref_ids, _ = _pad_trajectories(ref_samples)
        ref_traj_emb = _encode_traj_in_chunks(
            wrapper,
            motions=ref_motions,
            m_lens=ref_lens,
            chunk_size=int(args.embed_batch_size),
        )
        reference_embeddings = {sid: ref_traj_emb[i] for i, sid in enumerate(ref_ids)}
        report["reference"] = {
            "result_dir": str(ref_dir),
            "data_stats": ref_stats,
            "num_reference_embeddings": len(reference_embeddings),
        }

    for baseline_name, result_dir in baselines:
        log.info("Evaluating baseline '%s' from %s", baseline_name, result_dir)
        samples, stats = _load_baseline_samples(
            prompts=prompts,
            result_dir=result_dir,
            max_samples=None,
            strict_9d=bool(args.strict_9d),
            log=log,
        )

        baseline_entry: Dict[str, Any] = {
            "result_dir": str(result_dir),
            "data_stats": stats,
        }

        if len(samples) < int(args.min_eval_items):
            baseline_entry["status"] = "skipped_too_few_valid_items"
            log.warning(
                "Skip baseline '%s': only %d valid items (< min-eval-items=%d)",
                baseline_name,
                len(samples),
                int(args.min_eval_items),
            )
            report["baselines"][baseline_name] = baseline_entry
            continue

        eval_out = _evaluate_one_baseline(
            baseline_name=baseline_name,
            samples=samples,
            wrapper=wrapper,
            embed_batch_size=int(args.embed_batch_size),
        )
        baseline_entry["status"] = "ok"
        baseline_entry["metrics"] = eval_out["metrics"]

        if args.dump_per_sample:
            ps_dir = Path(args.dump_per_sample).expanduser().resolve()
            ps_dir.mkdir(parents=True, exist_ok=True)
            ps = eval_out["per_sample"]
            ps_path = ps_dir / f"{baseline_name}_per_sample.csv"
            with open(ps_path, "w", encoding="utf-8") as pf:
                pf.write("id,matching_score,t2m_rank,m2t_rank\n")
                for sid, ms, tr, mr in zip(
                    ps["id"], ps["matching_score"], ps["t2m_rank"], ps["m2t_rank"]
                ):
                    pf.write(f"{sid},{float(ms):.6f},{int(tr)},{int(mr)}\n")
            # sanity: per-sample mean must equal the aggregate matching_score
            ps_mean = float(np.mean(ps["matching_score"]))
            agg = float(eval_out["metrics"]["matching_score"])
            if abs(ps_mean - agg) > 1e-4:
                log.warning(
                    "[%s] per-sample mean %.6f != aggregate matching_score %.6f",
                    baseline_name, ps_mean, agg,
                )
            log.info("Wrote per-sample CSV to %s", ps_path)

        if reference_embeddings is not None:
            gen_ids = eval_out["sample_ids"]
            gen_emb = eval_out["traj_embeddings"]
            paired_gen = []
            paired_ref = []
            for i, sid in enumerate(gen_ids):
                ref = reference_embeddings.get(sid)
                if ref is None:
                    continue
                paired_gen.append(gen_emb[i])
                paired_ref.append(ref)
            if len(paired_gen) >= int(args.min_eval_items):
                paired_gen_arr = np.stack(paired_gen, axis=0)
                paired_ref_arr = np.stack(paired_ref, axis=0)
                baseline_entry["metrics"]["fid_vs_reference"] = float(
                    compute_fid(paired_ref_arr, paired_gen_arr)
                )
                baseline_entry["metrics"]["reference_diversity"] = float(
                    compute_diversity(paired_ref_arr, seed=0)
                )
                baseline_entry["metrics"]["paired_with_reference"] = int(len(paired_gen))
            else:
                baseline_entry["metrics"]["fid_vs_reference"] = float("nan")
                baseline_entry["metrics"]["reference_diversity"] = float("nan")
                baseline_entry["metrics"]["paired_with_reference"] = int(len(paired_gen))

        report["baselines"][baseline_name] = baseline_entry

        m = eval_out["metrics"]
        log.info(
            "[%s] n=%d | t2m R@1=%.2f R@3=%.2f R@5=%.2f | m2t R@1=%.2f | match=%.4f",
            baseline_name,
            m["num_evaluated"],
            m["t2m/R01"],
            m["t2m/R03"],
            m["t2m/R05"],
            m["m2t/R01"],
            m["matching_score"],
        )

    out_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (benchmark_root / "benchmark_clatr_eval.json").resolve()
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Wrote evaluation report to %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
