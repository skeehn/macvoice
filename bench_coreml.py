#!/usr/bin/env python3
"""CoreML engine benchmark — kokoro-coreml's Config-A path, Python runtime.

Same frozen texts and voice as the MLX baseline (bench.py TEXTS):
1 warmup + N timed synthesize() calls per bucket in one warm process,
fresh subprocess per bucket. Two layers reported:

- e2e (text -> audio): includes the PyTorch prefix (BERT/G2P/duration/F0Ntrain)
  + CoreML decode. Comparable 1:1 with benMLX bench protocol (both text-in).
- predict slice: time inside CompiledMLModel.predict (the CoreML part).
  Directionally comparable with their published Swift token-in table —
  but NOT the same protocol (Swift is token-in with a full-CoreML pipeline).

    source ~/macvoice/.venv-kcoreml/bin/activate   (requirements-coreml.txt)
    python bench_coreml.py [--runs 5] [--cu ALL] [--out data/benchmarks/coreml_baseline.json]

Pre-compile artifacts once: python scripts/fetch_coreml.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))  # for _coreml_runtime (imported inside RUNNER)

RUNNER = r'''
import json, os, sys, time

sys.path.insert(0, os.environ["MACVOICE_ROOT"])   # macvoice root: _coreml_runtime
mac_root = sys.argv[4]

key, runs, text, compute_unit = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[5]
sys.path.insert(0, os.path.join(mac_root, "kokoro-coreml"))

from _coreml_runtime import install_fast_models, predict_total
from coremltools.models.model import MLModel as _MLModel
from coremltools.models._compiled_model import CompiledMLModel as _Compiled
install_fast_models(_MLModel, _Compiled, os.path.join(mac_root, "coremlc"), compute_unit=compute_unit)
from kokoro.coreml_pipeline import HybridTTSPipeline

# Init + warmup untimed (covers pipeline init and first-predict ANE specialization).
pipe = HybridTTSPipeline(force_engine="coreml")
warm_start = time.perf_counter()
audio0, sr0 = pipe.synthesize(text, voice="af_heart", speed=1.0)
init_and_warmup_s = time.perf_counter() - warm_start

times, predict_warms = [], []
for _ in range(runs):
    before = predict_total[0]
    t0 = time.perf_counter()
    audio, sr = pipe.synthesize(text, voice="af_heart", speed=1.0)
    times.append(time.perf_counter() - t0)
    predict_warms.append(predict_total[0] - before)

import resource
print(json.dumps({
    "key": key, "init_and_warmup_s": round(init_and_warmup_s, 4),
    "times_s": [round(t, 4) for t in times],
    "predict_warm_s": [round(p, 4) for p in predict_warms],
    "audio_duration_s": round(len(audio) / sr, 3),
    "sample_rate": int(sr),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
}))
'''


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2


def _fmt(v):
    return " ".join("%.0f" % (x * 1000) + "ms" for x in v[:2])


def run_bucket(key, runs, text, cu):
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER, key, str(runs), text, str(REPO), cu],
        capture_output=True, text=True, cwd=str(REPO),
        env={**os.environ, "MACVOICE_ROOT": str(REPO)},
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    import importlib.metadata as md

    runs, cu = 5, "CPU_AND_GPU"
    argv = sys.argv[1:]
    if "--runs" in argv:
        runs = int(argv[argv.index("--runs") + 1])
    if "--cu" in argv:
        cu = argv[argv.index("--cu") + 1]
    out_path = "data/benchmarks/coreml_baseline.json"
    if "--out" in argv:
        out_path = argv[argv.index("--out") + 1]
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    from bench import TEXTS  # shared frozen inputs with the MLX bench

    doc = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip(),
            "memory_gb": round(int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                                  capture_output=True, text=True).stdout) / 2**30, 1),
            "python": sys.version.split()[0],
            "packages": {p: md.version(p) for p in
                         ("torch", "coremltools", "numpy", "transformers", "misaki")},
        },
        "engine": "kokoro-coreml HybridTTSPipeline(force_engine='coreml', compute_units=%s) "
                  "- Python Config-A: CoreML decoder-har-post buckets + PyTorch prefix" % cu,
        "compute_units": cu,
        "voice": "af_heart", "speed": 1.0,
        "protocol": "pipeline init + 1 warm synthesize() untimed, then %d timed runs "
                    "(median wall), fresh subprocess per bucket; MLModel.predict "
                    "time wrapped separately" % runs,
        "texts_source": "bench.py TEXTS (verbatim from kokoro-coreml bakeoff_harness.py)",
        "buckets": {},
    }
    for key in ("3s", "7s", "15s", "30s"):
        print(f"--- bucket {key} ({runs} runs) ---", flush=True)
        try:
            r = run_bucket(key, runs, TEXTS[key], cu)
        except RuntimeError as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            return 1
        times = r["times_s"]
        med = median(times)
        audio = r["audio_duration_s"]
        entry = {
            "init_and_warmup_s": r["init_and_warmup_s"],
            "runs_s": times,
            "median_ms": round(med * 1000, 1),
            "min_ms": round(min(times) * 1000, 1),
            "max_ms": round(max(times) * 1000, 1),
            "predict_warm_s": r["predict_warm_s"],
            "audio_duration_s": audio,
            "rtf_x_realtime": round(audio / med, 2),
            "peak_rss_mb": r["peak_rss_mb"],
        }
        doc["buckets"][key] = entry
        print(f"  warmup {r['init_and_warmup_s']:6.1f}s | median {entry['median_ms']:8.1f} ms | "
              f"predict {_fmt(entry['predict_warm_s'])} | audio {audio:5.1f}s | "
              f"{entry['rtf_x_realtime']:5.1f}x RT | RSS {entry['peak_rss_mb']:.0f} MB", flush=True)

    dest.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nsaved -> {dest}")


if __name__ == "__main__":
    sys.exit(main())
