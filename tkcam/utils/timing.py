"""
Reusable inference-timing helper, shared across all method inference scripts (different repos/envs).

Accurate GPU timing = cuda.synchronize() around the timed region + perf_counter; the first `warmup`
timed calls are discarded (CUDA context / cuDNN autotune / lazy weight init). Records per-call wall
time, the number of trajectories generated in that call (batch), and total frames, then writes a JSON
summary with both **latency** (s / trajectory) and **throughput** (trajectories / second), plus ms/frame.

Usage (in any inference loop):
    import sys; sys.path.insert(0, "/home/haozhe/CamTraj/Benchmark_hub")
    from timing_utils import Profiler
    prof = Profiler("TKCAM-noframe", warmup=3)
    for batch in batches:
        with prof.timed(n_traj=len(batch), n_frames=sum(lengths)):
            generate(batch)          # the core sampling call only
    prof.save("/path/timing_TKCAM_noframe.json", meta={"batch_size": bs, "device": dev})
"""
from __future__ import annotations
import json
import time
from contextlib import contextmanager

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None


def _sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


class Profiler:
    def __init__(self, name: str, warmup: int = 3):
        self.name = name
        self.warmup = warmup
        self.calls = []  # list of dicts: {dt, n_traj, n_frames}
        self._i = 0

    @contextmanager
    def timed(self, n_traj: int = 1, n_frames: int = 0):
        _sync()
        t0 = time.perf_counter()
        yield
        _sync()
        dt = time.perf_counter() - t0
        self._i += 1
        if self._i > self.warmup:                       # discard warm-up calls
            self.calls.append({"dt": dt, "n_traj": int(n_traj), "n_frames": int(n_frames)})

    def summary(self) -> dict:
        if not self.calls:
            return {"name": self.name, "n_calls": 0}
        import statistics as st
        total_time = sum(c["dt"] for c in self.calls)
        total_traj = sum(c["n_traj"] for c in self.calls)
        total_frames = sum(c["n_frames"] for c in self.calls)
        per_traj = [c["dt"] / max(c["n_traj"], 1) for c in self.calls]
        return {
            "name": self.name,
            "n_calls": len(self.calls),
            "n_traj": total_traj,
            "total_time_s": round(total_time, 4),
            "warmup_discarded": self.warmup,
            # latency: mean per-trajectory time within a call (meaningful at batch=1)
            "latency_s_per_traj_mean": round(st.mean(per_traj), 5),
            "latency_s_per_traj_std": round(st.pstdev(per_traj), 5) if len(per_traj) > 1 else 0.0,
            # throughput: trajectories per second across all timed calls (use native-batch run)
            "throughput_traj_per_s": round(total_traj / total_time, 4) if total_time > 0 else 0.0,
            "amortized_s_per_traj": round(total_time / max(total_traj, 1), 5),
            "ms_per_frame": round(1000.0 * total_time / total_frames, 4) if total_frames else None,
        }

    def save(self, path: str, meta: dict | None = None):
        out = self.summary()
        if meta:
            out["meta"] = meta
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        s = out
        print(f"[timing] {self.name}: n_traj={s.get('n_traj')} "
              f"latency={s.get('latency_s_per_traj_mean')}s/traj "
              f"throughput={s.get('throughput_traj_per_s')} traj/s "
              f"({s.get('ms_per_frame')} ms/frame) -> {path}")
        return out
