#!/usr/bin/env python3
"""Fetch kokoro-coreml's pinned artifacts and pre-compile them for this Mac.

Downloads the decoder-har-post buckets (3/7/15/30 s) from mattmireles/kokoro-coreml
at the SDK-pinned HF revision, then compiles each package once through Apple's
CoreML compiler so every later load skips the minutes-long ANE specialization
(macOS 26 beta quirk) and goes straight to predict.

    python scripts/fetch_coreml.py            # into <macvoice>/coremlc/ (~150 MB)

Then run macvoice with ~/macvoice/.venv-kcoreml/bin/python (requirements-coreml.txt).
"""

import shutil
import sys
import time
from pathlib import Path

REPO_ID = "mattmireles/kokoro-coreml"
REVISION = "034f7741bf938fd6b4b9c75c22bc72cd795f908a"  # SDK-pinned commit (their KokoroHostedModels.swift)
BUCKETS = (3, 7, 15, 30)


def main() -> int:
    macvoice_root = Path(__file__).resolve().parent.parent
    cache = macvoice_root / "coremlc"
    cache.mkdir(exist_ok=True)
    done = [cache / f"kokoro_decoder_har_post_{b}s.mlmodelc" for b in BUCKETS]
    if all(d.exists() for d in done):
        print("already fetched+compiled:", cache)
        return 0
    try:
        from huggingface_hub import snapshot_download
        import coremltools as ct
    except ImportError:
        print("pip install huggingface_hub coremltools  (see requirements-coreml.txt)", file=sys.stderr)
        return 1

    root = cache.parent
    target = root / "kokoro-coreml"
    target.mkdir(exist_ok=True)
    patterns = [f"coreml/kokoro_decoder_har_post_{b}s.mlpackage/**" for b in BUCKETS]
    print(f"downloading {len(BUCKETS)} packages from {REPO_ID}@{REVISION[:12]} …", flush=True)
    snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=str(target),
                      allow_patterns=patterns)

    for b in BUCKETS:
        pkg = target / "coreml" / f"kokoro_decoder_har_post_{b}s.mlpackage"
        dest = cache / f"kokoro_decoder_har_post_{b}s.mlmodelc"
        if dest.exists():
            print(f"  {b}s: already compiled"); continue
        if not pkg.exists():
            print(f"  {b}s: package missing after download", file=sys.stderr); return 1
        t0 = time.time()
        tmp = ct.models.utils.compile_model(str(pkg))
        shutil.move(tmp, str(dest))
        print(f"  {b}s compiled -> {dest.name} in {time.time() - t0:.1f}s", flush=True)
    print(f"\nDone. Artifacts: {cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
