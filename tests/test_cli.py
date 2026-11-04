"""macvoice CLI unit tests — no model downloads, no engine needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from macvoice import TEXTS, combine_wavs, detect_engine, main  # noqa: E402

import wave  # noqa: E402


def make_chunk(tmp_path, n_zeros, i=0):
    import array

    p = tmp_path / f"out_{i}.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        frames = array.array("h", [0] * n_zeros).tobytes()
        w.writeframes(frames)
    return p


def test_texts_match_kokoro_frozen_inputs(tmp_path):
    """The 3s text is verbatim from kokoro-coreml's bakeoff harness."""
    assert TEXTS["3s"] == "The quick brown fox jumps over the dog."
    assert 25 < len(TEXTS["30s"].split()) < 90


def test_combine_wavs_concatenates_chunks(tmp_path):
    c0, c1 = make_chunk(tmp_path, 24000, 0), make_chunk(tmp_path, 12000, 1)
    out = tmp_path / "joined.wav"
    combine_wavs([c0, c1], out)
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() == 36000
        assert w.getframerate() == 24000


def test_benchmark_flag_rejects_bad_engine():
    """`--benchmark --engine bogus` must fail fast at argparse, never launch benches."""
    import subprocess

    cli = str(Path(__file__).resolve().parent.parent / "macvoice.py")
    out = subprocess.run([sys.executable, cli, "--benchmark", "--engine", "bogus"],
                         capture_output=True, text=True, cwd=str(Path(cli).parent))
    assert out.returncode == 2
    assert "invalid choice" in (out.stderr + out.stdout)


def test_help_exits_zero():
    import subprocess

    cli = str(Path(__file__).resolve().parent.parent / "macvoice.py")
    out = subprocess.run([sys.executable, cli], capture_output=True, text=True)
    assert out.returncode == 2  # no text -> usage
    assert "offline tts" in out.stdout.lower()
