# GPU burst budget

`solution/gpu_budget.py` is a read-only calculator. It never launches or
stops processes. Measure the peak VRAM of one real worker using the same model,
batch, precision and input size before increasing concurrency.

```bash
python solution/gpu_budget.py --per-worker-gib 6 --reserve-gib 2
```

On a genuinely empty A100 80 GiB, the theoretical upper bound at 6 GiB per
worker is `floor((80 - 2) / 6) = 13` workers per GPU. This is not a launch
guarantee: allocator fragmentation and transient memory require margin.

On a shared GPU, use current free VRAM only as a capacity estimate and do not
take over another user's process. Probe immediately before launch and keep
the 2 GiB reserve (or a larger scheduler margin).
