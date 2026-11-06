# macvoice

Offline TTS (Kokoro-82M) for Apple Silicon: **one CLI, two engines**. MLX is
the always-works pip install; CoreML runs the precompiled kokoro-coreml
pipeline on the ANE/GPU and beats MLX at every bucket on the same hardware.
One ~250-line Python file, stdlib + engine deps only.

```sh
pip install macvoice[mlx]        # engine: MLX (works everywhere)
macvoice "Hello from my Mac."    # speaks through the speakers
macvoice -o hi.wav "Save this."  # writes 24 kHz mono WAV
```

Two engines under one roof, auto-selected:

| engine | backend | works | notes |
|---|---|---|---|
| `mlx` *(default)* | [mlx-audio](https://github.com/mlx-community/mlx-audio) `Kokoro-82M-bf16` | everywhere | pip only: `pip install macvoice[mlx]` |
| `coreml` | [kokoro-coreml](https://github.com/mattmireles/kokoro-coreml) decoder-har-post buckets | with artifacts | Apple Neural Engine / GPU; `python scripts/fetch_coreml.py` |

## Benchmark — Apple M3 MacBook Air 16 GB (Oct 31 – Nov 2, 2026)

Same frozen texts, voice (`af_heart`), speed 1.0, model, and protocol as
kokoro-coreml's bakeoff: 1 warmup + 5 timed runs, **median end-to-end wall
time (text → audio)**, fresh process per bucket. Raw records in
`data/benchmarks/*.json`.

| bucket (audio) | macvoice `mlx` | macvoice `coreml` (ANE) | macvoice `coreml` (GPU) | fastest macvoice | kokoro-coreml published CoreML* |
|---|---|---|---|---|---|
| 3s (2.8s) | 1,130 ms | 220 ms | **169 ms** | **169 ms** (6.7×) | 62–98 ms (token-in) |
| 7s (6.8s) | 1,501 ms | **328 ms** (20.6× RT) | 350 ms | **328 ms** (4.6×) | 54–145 ms (token-in) |
| 15s (13.9s) | 1,998 ms | **724 ms** (19.2× RT) | 751 ms | **724 ms** (2.8×) | 89–261 ms (token-in) |
| 30s (27.4s) | 3,341 ms | 2,784 ms | **2,026 ms** | **2,026 ms** (1.6×) | 159–480 ms (token-in) |

Peak RSS @30s: 880 MB (mlx) · 1,422–1,463 MB (coreml). Speedups are fastest
macvoice vs our mlx on the identical machine/protocol.

\* Not the same protocol: their published table is the **Swift runtime with
token-IDs in** (no G2P, no BERT). Our columns are **end-to-end from text** on
one machine — the honest engine-vs-engine comparison. Their MLX cross-refs on
M-series hardware (e.g. M2 Air 4,797–5,187 ms @ 27.4 s) bracket our M3 MLX
baseline. Their 3s clip regression on Ultra-class ANE routing recurs here too
(ANE is *slower* than GPU on 3s clips; faster on 7–15s). Numbers were not
recorded on a quiet host.

## Why ANE wins at 7–15s and loses at 3s/30s (v0.1.1 placement study)

`scripts/compute_plan.py` loads coremltools 8.3 `MLComputePlan` on our
precompiled buckets in both compute-unit modes and records, per operator, the
runtime's preferred compute device (ANE/GPU/CPU) plus its **estimated**
workload share (`MLComputePlanCost.weight`, 0–1). Raw records:
`data/benchmarks/compute_plan.json`.

| bucket | `ALL`-mode placement (est. workload) | `ALL` med | `CPU_AND_GPU` med | winner | why |
|---|---|---|---|---|---|
| 3s | **81% ANE** / 19% GPU | 220 ms | 169 ms | GPU | ANE grabs nearly everything (327/328 muls, 218/218 adds, 45/51 convs); tensors are tiny, so ANE's fixed per-op dispatch overhead never amortizes — one device running 100% of the graph wins |
| 7s | 14% ANE / **86% GPU** | 328 ms | 350 ms | ANE (hybrid) | the plan itself pushes the heavy conv stack back to GPU and keeps a thin ANE slice; the hybrid edges out GPU-only by ~7% |
| 15s | 15% ANE / **85% GPU** | 724 ms | 751 ms | ANE (hybrid) | same rebalanced hybrid; and per op-kind it splits *within* a kind (mul: 163 ANE + 165 GPU) instead of taking whole kinds like at 3s |
| 30s | **9.9% ANE / 90.1% GPU** (61 ANE / 1,113 GPU ops; even `ios16.mul` splits 306 GPU / 22 ANE) | 2,784 ms | 2,026 ms | GPU | the runtime itself keeps only a ~10% ANE slice here, so `ALL` ≈ GPU + sync overhead — the thin ANE slice can't pay for itself. *(Cold plan load: ~16.5 min on a fresh (respawned) daemon, cached thereafter; earlier we only ever saw the wedge signatures of that cold load: daemon 100% CPU, or client parked at 0% for ~1.5 min with an idle daemon.)* |

Findings:

- `CPU_AND_GPU` is a clean single-device baseline in every bucket: all 1,174
  non-constant ops prefer GPU, est. workload 100% GPU, placement identical
  across buckets and deterministic.
- `ALL`'s placement is **bucket-size dependent and monotonically collapsing**:
  ANE workload share 81% (3s) → 14% (7s) → 15% (15s) → 9.9% (30s) — it
  overcommits to ANE on the smallest graph and rebalances increasingly
  GPU-heavy as tensors grow. At 7–15s the hybrid margin beats GPU-only by
  ~3–4%; by 30s the thin ANE slice loses to GPU-only. That's exactly our
  `coreml` (ANE) column vs `coreml` (GPU) column.
- The iSTFT tail never leaves the GPU: `conv_transpose` is GPU-preferred under
  `ALL` in all buckets (~10–15% of est. workload), and at 3s an additional 6
  convs stay GPU — few ops, but they carry the vocoder tail.
- Cold plan-generation cost is **superlinear in bucket size** (fresh daemon,
  no cache): 0.2s (3s) → 19.5s (7s) → 52.6s (15s) → **993.8s (30s)** — this
  is the same ANECompilerService machinery the macvoice placement study
  reverse-engineered (`data/benchmarks/ANECOMPILER_NOTES.md`): no time-based
  guard anywhere, cost dominated by the CP-allocator/scheduler stages. Every
  bucket is instant (~0.2–0.4s) once cached.
- Bookkeeping ops (`slice_by_index`, 6–12 stragglers) prefer CPU; the 1,079
  constants in each graph are unbound and weightless.
- Caveat: `MLComputePlanCost.weight` is the runtime's *own* cost model — it
  clearly undersells ANE's fixed dispatch/compile overhead at small sizes
  (the 3s row). Placement is exact; the weights are an estimate.

## v0.1.1

Placement study release; no engine code changed.

- `scripts/compute_plan.py` — per-op compute-unit placement via coremltools
  8.3 `MLComputePlan` (ALL vs CPU_AND_GPU) on the precompiled buckets, with
  per-op estimated workload shares; data in
  `data/benchmarks/compute_plan.json`.
- README section (above) explaining the ANE-vs-GPU medians per bucket from
  that placement data.
- Documented the 30s `ALL`-mode plan-generation wedge (`ANECompilerService`,
  macOS 26) and the `--no-costs` escape hatch in the study script.

## v0.1.0

First release of the tool.

- `macvoice` CLI: engines (mlx + coreml) under one roof, autodetect, `-o file.wav`.
- Deterministic bench (mlx/coreml) matching kokoro-coreml's frozen texts and
  protocol; first-party M3 Air baselines recorded in `data/benchmarks/`.
- `scripts/fetch_coreml.py` — one-command CoreML artifact acquisition.

## Why it's fast (and citation)

Kokoro's vocoder is conv-heavy — the ANE eats it — and kokoro-coreml cuts the
model at its seams. We run their Python Config-A path (CoreML
decoder-har-post buckets) instead of trying to out-optimize MLX. Beats MLX at
every bucket on the same hardware; thanks to
[mattmireles/kokoro-coreml](https://github.com/mattmireles/kokoro-coreml)
for the artifacts, pipeline, and protocols (Apache-2.0).

## CoreML engine install

```sh
git clone --depth 1 https://github.com/mattmireles/kokoro-coreml ~/macvoice/kokoro-coreml   # OR: use scripts/fetch_coreml.py's clone
python -m venv ~/macvoice/.venv-kcoreml
~/macvoice/.venv-kcoreml/bin/pip install -r ~/macvoice/requirements-coreml.txt
~/macvoice/.venv-kcoreml/bin/python scripts/fetch_coreml.py    # downloads (~150 MB) + precompiles
~/macvoice/.venv-kcoreml/bin/python macvoice.py "CoreML engine ready."
```

macOS note: on macOS 26.0 the first ANE load of each package runs through
Apple's ANECompilerService and can take ~1 min per package (or wedge the
daemon under repeated loads — `sudo pkill -9 ANECompilerService` clears it;
it respawns cleanly). `scripts/fetch_coreml.py` pre-compiles so you pay once,
forever.

## `macvoice bench`

```sh
macvoice bench --engine both --runs 5    # writes data/benchmarks/*.json
```

Protocol identical to kokoro-coreml's bakeoff (frozen texts, 1 warmup + 5
timed medians, fresh subprocess per bucket); also records API predict-stage
time separately for CoreML.

## License

MIT. Model weights: Kokoro-82M (Apache-2.0, hexgrad). MacOS synthesizer
artifacts: Apache-2.0 (mattmireles/kokoro-coreml).
