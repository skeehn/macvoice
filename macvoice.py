#!/usr/bin/env python3
"""macvoice — one-liner offline TTS for Apple Silicon, two engines under one roof.

    macvoice "Hello from my Mac."                 # speaks through speakers
    macvoice -o out.wav "Read this later."        # writes 24 kHz mono WAV
    macvoice --engine mlx "..."                   # force an engine
    macvoice bench [--engine both] [--runs 5]     # deterministic benchmark

Engine autodetect: CoreML artifacts pre-compiled under coremlc/ -> 'coreml'
when its deps import; otherwise 'mlx' (mlx-audio, always installable).

  mlx    — mlx-community/Kokoro-82M-bf16 via mlx-audio; text -> wav end to end.
  coreml — kokoro-coreml's Config-A path (decoder-har-post buckets on ANE) +
           the PyTorch prefix. Artifacts via scripts/fetch_coreml.py; deps in
           requirements-coreml.txt; install target: ~/macvoice/.venv-kcoreml.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent
COMPILE_CACHE = REPO / "coremlc"
KCOREML_REPO = REPO / "kokoro-coreml"
MODEL_MLX = "mlx-community/Kokoro-82M-bf16"
VOICE, SPEED, LANG = "af_heart", 1.0, "a"

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


# --- wav helpers -------------------------------------------------------------

def write_wav(path: Path, audio, sr: int) -> None:
    """float samples (any shape) -> 16-bit mono PCM wav."""
    import numpy as np

    a = np.asarray(audio, dtype="float32").squeeze()
    if a.size and float(np.max(np.abs(a))) > 0:
        a = a / max(1e-7, float(np.max(np.abs(a))))
    data = (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(data)
    print(f"wrote {path} ({a.size / sr:.1f}s @ {int(sr)} Hz)")


def combine_wavs(chunks, out: Path) -> None:
    """mlx-audio writes <dir>/<prefix>_<n>.wav chunks; concat frames into one."""
    if len(chunks) == 1:
        shutil.copyfile(chunks[0], out)
        return
    frames, sr = [], None
    for c in sorted(chunks):
        with wave.open(str(c), "rb") as w:
            if sr is None:
                sr = w.getframerate()
            frames.append(w.readframes(w.getnframes()))
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(frames))


def play_file(path: Path) -> None:
    subprocess.run(["afplay", str(path)], check=False)


# --- engines -----------------------------------------------------------------

def say_mlx(text: str, out: Path | None, voice: str, model: str) -> None:
    """Synthesize with mlx-audio. Deps: .venv (mlx, mlx-audio, misaki[en])."""
    try:
        from mlx_audio.tts.generate import generate_audio
    except ImportError:
        sys.exit("macvoice: mlx deps missing — reinstall with `pip install macvoice[mlx]`")

    if out is None:  # speak: render then hand the file to afplay
        with tempfile.TemporaryDirectory() as td:
            generate_audio(text, model=model, voice=voice, lang_code=LANG,
                           speed=SPEED, audio_format="wav",
                           output_path=td, file_prefix="out", play=False)
            chunks = sorted(Path(td).glob("out*.wav"))
            combined = Path(td) / "play.wav"
            combine_wavs(chunks, combined)
            play_file(combined)
        return
    with tempfile.TemporaryDirectory() as td:
        generate_audio(text, model=model, voice=voice, lang_code=LANG,
                       speed=SPEED, audio_format="wav",
                       output_path=td, file_prefix="out", play=False)
        combine_wavs(sorted(Path(td).glob("out*.wav")), out)
        print(f"wrote {out}")


def say_coreml(text: str, out: Path | None, voice: str) -> None:
    """Synthesize via kokoro-coreml's precompiled pipeline. Deps: .venv-kcoreml."""
    try:
        sys.path.insert(0, str(KCOREML_REPO))
        sys.path.insert(0, str(REPO))
        from coremltools.models.model import MLModel as _MLModel
        from coremltools.models._compiled_model import CompiledMLModel as _Compiled
        from _coreml_runtime import install_fast_models, COMPILE_CACHE
    except ImportError:
        sys.exit("macvoice: coreml deps missing — use the pinned env: "
                 f"{REPO / '.venv-kcoreml' / 'bin' / 'python'} (see requirements-coreml.txt)")

    install_fast_models(_MLModel, _Compiled, str(COMPILE_CACHE), compute_unit="ALL")
    from kokoro.coreml_pipeline import HybridTTSPipeline

    pipe = HybridTTSPipeline(force_engine="coreml")
    audio, sr = pipe.synthesize(text, voice=voice, speed=SPEED)
    if out is None:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "play.wav"
            write_wav(wav, audio, sr)
            play_file(wav)
        return
    write_wav(out, audio, sr)


def detect_engine() -> str:
    """coreml artifacts + deps -> coreml; else mlx; else explain how to install."""
    have_coreml = (COMPILE_CACHE.is_dir() and any(COMPILE_CACHE.glob("*.mlmodelc"))
                   and KCOREML_REPO.is_dir()
                   and importlib.util.find_spec("coremltools") is not None
                   and importlib.util.find_spec("torch") is not None)
    if have_coreml:
        return "coreml"
    try:
        import mlx_audio  # noqa: F401
        return "mlx"
    except ImportError:
        pass
    sys.exit("macvoice: no engine. coreml: scripts/fetch_coreml.py + requirements-coreml.txt; "
             "or `pip install macvoice[mlx]` for mlx.")


# --- bench (subprocesses; heavy deps are venv-scoped) ------------------------

BENCHJOBS = {  # engine -> (venv dir, script)
    "mlx": (".venv", "bench.py"),
    "coreml": (".venv-kcoreml", "bench_coreml.py"),
}


def run_bench(engines, runs: int) -> int:
    rc = 0
    for eng in engines:
        if eng not in BENCHJOBS:
            continue
        venv, script = BENCHJOBS[eng]
        py = REPO / venv / "bin" / "python"
        interp = str(py) if py.exists() else sys.executable
        cmd = [interp, str(REPO / script), "--runs", str(runs)]
        if eng == "coreml":
            cmd += ["--cu", "ALL"]
        print(f"\n=== macvoice bench [{eng}] ===", flush=True)
        if subprocess.run(cmd, cwd=REPO).returncode != 0:
            rc = 1
    return rc


# --- CLI ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="macvoice", description="Offline TTS (Kokoro-82M) for Apple Silicon — "
                                     "mlx always works, coreml flies when artifacts are present.")
    ap.add_argument("text", nargs="?", help="text to speak")
    ap.add_argument("-o", "--out", metavar="FILE.wav", help="write WAV here instead of playing")
    ap.add_argument("--engine", choices=["mlx", "coreml"], help="engine override (default: autodetect)")
    ap.add_argument("--voice", default=VOICE, help=f"voice pack (default: {VOICE})")
    ap.add_argument("--model", default=MODEL_MLX, help="mlx model override")
    ap.add_argument("--benchmark", action="store_true",
                    help="run the deterministic 3s/7s/15s/30s benchmark")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per bucket for --benchmark")
    args = ap.parse_args(argv)

    if args.benchmark:
        engines = ["coreml", "mlx"] if not args.engine else [args.engine]
        return run_bench(engines, args.runs)

    if not args.text:
        ap.print_help()
        return 2

    engine = args.engine or detect_engine()
    t0 = time.perf_counter()
    if engine == "coreml":
        say_coreml(args.text, Path(args.out) if args.out else None, args.voice)
    else:
        say_mlx(args.text, Path(args.out) if args.out else None, args.voice, args.model)
    if not args.out:
        print(f"[{engine} took {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
