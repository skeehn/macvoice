#!/usr/bin/env python3
"""macvoice bench — deterministic Kokoro TTS benchmark on Apple Silicon.

Uses the exact frozen texts published by kokoro-coreml's bakeoff harness
(3s / 7s / 15s / 30s buckets, voice=af_heart, speed=1.0) so numbers are
comparable against their published MLX/CoreML tables across M-series chips.

Protocol per bucket: 1 warmup + N timed runs, fresh subprocess per bucket
(so RSS is a real process peak, not accumulated). Reports median/min/max
wall time end-to-end (text -> wav on disk), RTF, and peak RSS.

    python bench.py [--runs 5] [--out data/benchmarks/mlx_baseline.json]
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

# --- Frozen benchmark texts (verbatim from kokoro-coreml bakeoff_harness.py) ---
TEXTS = {
    "3s": "The quick brown fox jumps over the dog.",
    "7s": (
        "The morning sun cast long shadows across the garden as birds began "
        "their chorus in the ancient oak tree."
    ),
    "15s": (
        "The ancient lighthouse stood alone on the rocky cliff, its beam sweeping "
        "across dark waters with the patience of centuries. Ships had come and gone, "
        "storms had battered its walls, yet still it turned, guiding sailors home."
    ),
    "30s": (
        "When the last train departed that evening, the platform fell silent. "
        "The old stationmaster locked the ticket office with trembling hands, "
        "running his fingers along the worn counter where countless journeys "
        "had begun. Outside, autumn wind scattered golden leaves across the "
        "empty tracks. He had spent forty years here, watching the world rush "
        "past in a blur of faces and farewells. The station would stand a while "
        "longer, its clock still ticking, its roof sheltering the pigeons."
    ),
}
MODEL = "mlx-community/Kokoro-82M-bf16"
VOICE, SPEED, LANG = "af_heart", 1.0, "a"

RUNNER = r'''
import json, resource, sys, time
from pathlib import Path
from wave import open as wopen

def one(text, outdir, key):
    from mlx_audio.tts.generate import generate_audio
    t0 = time.perf_counter()
    generate_audio(text, model=%r, voice=%r, lang_code=%r, speed=%.1f,
                   audio_format="wav", output_path=outdir, file_prefix=key)
    return time.perf_counter() - t0

key, outdir, runs, text = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
one(text, outdir, key)  # warmup — untimed
times = [one(text, outdir, key) for _ in range(runs)]
files = sorted(Path(outdir).glob(key + "*.wav")) or sorted(Path(outdir).glob("*.wav"))
dur = 0.0
for fp in files:
    with wopen(str(fp)) as w:
        dur += w.getnframes() / w.getframerate()
print(json.dumps({"key": key, "times_s": times, "audio_duration_s": dur,
                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}))
''' % (MODEL, VOICE, LANG, SPEED)

import json
import importlib.metadata as md  # (kept late; RUNNER child imports json itself)

def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2


def run_bucket(key, runs, tmpdir):
    import shutil
    outdir = tmpdir / key
    shutil.rmtree(outdir, ignore_errors=True)  # fresh dir each bucket
    text = TEXTS[key]
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER, key, str(outdir), str(runs), text],
        capture_output=True, text=True, cwd=str(Path.cwd()),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1200:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def machine_info():
    return {
        "platform": platform.platform(),
        "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip(),
        "memory_gb": round(int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                              capture_output=True, text=True).stdout) / 2**30, 1),
        "python": platform.python_version(),
    }


def main():
    runs = 5
    argv = sys.argv[1:]
    if "--runs" in argv:
        runs = int(argv[argv.index("--runs") + 1])
    out_path = "data/benchmarks/mlx_baseline.json"
    if "--out" in argv:
        out_path = argv[argv.index("--out") + 1]
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = dest.parent / "tmp_wavs"
    tmpdir.mkdir(exist_ok=True)

    import importlib.metadata as md
    doc = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": machine_info(),
        "packages": {p: md.version(p) for p in ("mlx", "mlx-audio", "misaki")},
        "model": MODEL, "voice": VOICE, "speed": SPEED,
        "protocol": "1 warmup + %d timed runs per bucket, fresh subprocess per bucket, "
                    "median of end-to-end text->wav wall time" % runs,
        "texts_source": "kokoro-coreml bakeoff_harness.py (frozen constants)",
        "buckets": {},
    }
    for key in ("3s", "7s", "15s", "30s"):
        print(f"--- bucket {key} ({runs} runs) ---", flush=True)
        r = run_bucket(key, runs, tmpdir)
        times = r["times_s"]
        med = median(times)
        rtf = med / r["audio_duration_s"]
        doc["buckets"][key] = {
            "runs_s": [round(t, 4) for t in times],
            "median_ms": round(med * 1000, 1),
            "min_ms": round(min(times) * 1000, 1),
            "max_ms": round(max(times) * 1000, 1),
            "audio_duration_s": round(r["audio_duration_s"], 3),
            "rtf_x_realtime": round(1 / rtf, 2) if rtf else None,
            "peak_rss_mb": round(r["peak_rss_mb"], 1),
        }
        b = doc["buckets"][key]
        print(f"  median {b['median_ms']:7.1f} ms  | audio {b['audio_duration_s']:5.1f}s  "
              f"| {b['rtf_x_realtime']:6.1f}x real-time  | peak RSS {b['peak_rss_mb']:.0f} MB")

    dest.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nsaved -> {dest}")
    (tmpdir / ".gitkeep").write_text("")


if __name__ == "__main__":
    main()
