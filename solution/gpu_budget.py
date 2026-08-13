#!/usr/bin/env python3
"""Read-only VRAM budget calculator for parallel AIC workers.

The helper never launches, stops, or reconfigures a process. It computes an
upper bound from current free VRAM (or total VRAM for a cold server) while
keeping an explicit reserve on every selected GPU.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess


def query_gpus():
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi is not available")
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        index, name, total, used, free = [x.strip() for x in line.split(",", 4)]
        rows.append({
            "index": int(index), "name": name,
            "total_mib": int(float(total)), "used_mib": int(float(used)),
            "free_mib": int(float(free)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-worker-gib", type=float, required=True,
                    help="empirical peak VRAM per independent worker/process")
    ap.add_argument("--reserve-gib", type=float, default=2.0,
                    help="VRAM left unused on each GPU (default: 2)")
    ap.add_argument("--gpu", type=int, action="append",
                    help="GPU index to show; repeat for multiple GPUs")
    ap.add_argument("--from-total", action="store_true",
                    help="plan a cold/free server from total VRAM")
    ap.add_argument("--target-workers", type=int,
                    help="also report the largest peak allowed for this count")
    args = ap.parse_args()
    if args.per_worker_gib <= 0 or args.reserve_gib < 0:
        ap.error("--per-worker-gib must be > 0 and --reserve-gib must be >= 0")
    if args.target_workers is not None and args.target_workers <= 0:
        ap.error("--target-workers must be > 0")

    reserve = args.reserve_gib * 1024
    worker = args.per_worker_gib * 1024
    selected = set(args.gpu) if args.gpu else None
    for row in query_gpus():
        if selected is not None and row["index"] not in selected:
            continue
        basis = row["total_mib"] if args.from_total else row["free_mib"]
        usable = max(0.0, basis - reserve)
        workers = math.floor(usable / worker)
        headroom = max(0.0, basis - workers * worker)
        basis_name = "total" if args.from_total else "free"
        print(
            f"GPU {row['index']} {row['name']}: {basis_name}={basis/1024:.2f} GiB, "
            f"reserve={args.reserve_gib:.2f} GiB, usable={usable/1024:.2f} GiB, "
            f"max_workers={workers} at {args.per_worker_gib:.2f} GiB/worker, "
            f"headroom_after_max={headroom/1024:.2f} GiB"
        )
        if args.target_workers is not None:
            peak = usable / args.target_workers / 1024
            print(
                f"  target_workers={args.target_workers}: "
                f"allowed_peak_per_worker={peak:.2f} GiB "
                f"(reserve remains {args.reserve_gib:.2f} GiB)"
            )


if __name__ == "__main__":
    main()
