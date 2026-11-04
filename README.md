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
