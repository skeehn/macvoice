"""CoreML fast-load runtime shared by bench_coreml.py and macvoice.py.

macOS 26.0 beta quirk: loading a .mlpackage re-runs ANE specialization each
process (minutes per package through ANECompilerService); pre-compiling each
package once to .mlmodelc (via ct.models.utils.compile_model, which bundles
Apple's compiler API) and loading *that* skips it entirely. This module gives
their HybridTTSPipeline a drop-in model object that predicts from the compiled
archive while get_spec() serves the protobuf of the original package.
"""

import os
import time
from pathlib import Path

MACVOICE_ROOT = Path(__file__).resolve().parent
COMPILE_CACHE = str(MACVOICE_ROOT / "coremlc")
predict_total = [0.0]  # seconds spent inside CompiledMLModel.predict


def install_fast_models(mlmodel_cls, compiled_cls, cache_dir=None, compute_unit=None):
    """Make ct.models.MLModel load .mlmodelc from cache_dir when it exists.

    Returns the replacement class. The pipeline only needs .predict() and
    .get_spec(); both are provided. predict() time accumulates into
    ``predict_total[0]``.
    """
    cache_dir = cache_dir or COMPILE_CACHE
    spec_cache = {}

    class FastModel:
        def __init__(self, path, *a, **kw):
            cand = None
            if isinstance(path, str) and path.endswith(".mlpackage"):
                name = os.path.basename(path)[: -len(".mlpackage")] + ".mlmodelc"
                c = os.path.join(cache_dir, name)
                if os.path.isdir(c):
                    cand = c
            if cand is not None:
                if compute_unit is not None:
                    import coremltools as _ctu
                    kw["compute_units"] = getattr(_ctu.ComputeUnit, compute_unit)
                self._m = compiled_cls(cand, **kw)
                if path not in spec_cache:
                    import coremltools as ct
                    spec_cache[path] = ct.utils.load_spec(path)
                self._cached_spec = spec_cache[path]
            else:
                self._m = mlmodel_cls(path, *a, **kw)
                self._cached_spec = None

        def predict(self, *a, **kw):
            t0 = time.perf_counter()
            out = self._m.predict(*a, **kw)
            predict_total[0] += time.perf_counter() - t0
            return out

        def get_spec(self):
            return self._cached_spec if self._cached_spec is not None else self._m.get_spec()

    import coremltools as ct
    ct.models.MLModel = FastModel
    return FastModel
