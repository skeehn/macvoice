#!/usr/bin/env python3
"""Per-op compute-unit placement study for the CoreML decoder buckets (macvoice v0.1.1).

Loads an MLComputePlan (coremltools 8.3) on each precompiled .mlmodelc in coremlc/
under two compute-unit modes (ALL vs CPU_AND_GPU) and records:
  - which operator names the runtime *prefers* to place on ANE / GPU / CPU,
  - the estimated workload share per device (MLComputePlanCost.weight, 0..1),
  - how placement changes between the two modes.

That placement + workload share is what explains our medians: why ANE beats GPU
at 7-15s but loses at 3s and 30s. Output: data/benchmarks/compute_plan.json.

Cost estimation (and any load that must consult ANECompilerService) can wedge
root's ANECompilerService at 100% CPU on macOS 26 for large buckets; run with
--no-costs and/or bucket/mode subsets, then re-run the remainder to merge.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import coremltools as ct
from coremltools.models import compute_plan as cp

ROOT = Path(__file__).resolve().parent.parent
COREMLC = ROOT / "coremlc"
OUT = ROOT / "data" / "benchmarks" / "compute_plan.json"
BUCKETS = ("3s", "7s", "15s", "30s")
MODE_NAMES = ("ALL", "CPU_AND_GPU")
BENCH_FILE = {"ALL": "coreml_ane.json", "CPU_AND_GPU": "coreml_baseline.json"}


def label(dev) -> str:
    n = type(dev).__name__
    if "NeuralEngine" in n:
        return "ANE"
    if "GPU" in n:
        return "GPU"
    if "CPU" in n:
        return "CPU"
    return n


def machine() -> dict:
    cpu = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    ).stdout.strip()
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
    return {
        "cpu": cpu or platform.processor(),
        "memory_gb": round(int(mem) / (1 << 30), 1) if mem.strip().isdigit() else None,
        "os": platform.mac_ver()[0],
        "python": platform.python_version(),
        "coremltools": ct.__version__,
    }


def load_bench_medians() -> dict:
    out = {}
    for mode, fname in BENCH_FILE.items():
        path = ROOT / "data" / "benchmarks" / fname
        if not path.exists():
            continue
        buckets = json.loads(path.read_text()).get("buckets", {})
        for b, rec in buckets.items():
            out.setdefault(b, {})[mode] = rec.get("median_ms")
    return {b: out[b] for b in BUCKETS if b in out}


def scan(plan, ops, with_costs: bool) -> dict:
    op_counts, device_ops, by_op, supported = Counter(), Counter(), {}, {}
    weight_by_device, weight_by_op_dev = defaultdict(float), defaultdict(float)
    supported_multi = 0
    for op in ops:
        name = op.operator_name
        op_counts[name] += 1
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = label(usage.preferred_compute_device) if usage is not None else "unbound"
        device_ops[dev] += 1
        by_op.setdefault(name, Counter())[dev] += 1
        if usage is not None:
            devs = {label(d) for d in usage.supported_compute_devices}
            supported.setdefault(name, set()).update(devs)
            supported_multi += len(devs) > 1
            cost = plan.get_estimated_cost_for_mlprogram_operation(op) if with_costs else None
            if cost is not None:
                weight_by_device[dev] += cost.weight
                weight_by_op_dev[name, dev] += cost.weight
    return {
        "ops_total": len(ops),
        "cost_estimation": with_costs,
        "op_counts": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
        "device_ops": dict(sorted(device_ops.items())),
        "by_op": {k: dict(v) for k, v in sorted(by_op.items())},
        "est_weight_total": round(sum(weight_by_device.values()), 4) if with_costs else None,
        "est_weight_by_device": (
            {k: round(v, 4) for k, v in sorted(weight_by_device.items())} if with_costs else None
        ),
        "est_weight_by_op": (
            {
                f"{op}@{dev}": round(w, 5)
                for (op, dev), w in sorted(weight_by_op_dev.items(), key=lambda kv: -kv[1])
                if w >= 0.001
            }
            if with_costs
            else None
        ),
        "supported_union_by_op": {k: sorted(v) for k, v in sorted(supported.items())},
        "ops_supported_by_multiple_devices": supported_multi,
    }


def parse_args(argv):
    ap = argparse.ArgumentParser(description="MLComputePlan placement study (macvoice)")
    ap.add_argument("--buckets", default=",".join(BUCKETS), help="comma subset of 3s,7s,15s,30s")
    ap.add_argument("--modes", default=",".join(MODE_NAMES), help="comma subset of ALL,CPU_AND_GPU")
    ap.add_argument("--no-costs", action="store_true", help="skip MLComputePlanCost estimation")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    buckets = tuple(b.strip() for b in args.buckets.split(",") if b.strip())
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine": machine(),
        "source": "MLComputePlan on precompiled coremlc/kokoro_decoder_har_post_*s.mlmodelc",
        "protocol": "deterministic placement enumeration (not a timing bench); "
        "preferred_compute_device per op + optional MLComputePlanCost.weight sums",
        "bench_median_ms": load_bench_medians(),
        "modes": {},
    }
    for mode_name in modes:
        mode = getattr(ct.ComputeUnit, mode_name)
        result["modes"][mode_name] = {}
        for bucket in buckets:
            t0 = time.perf_counter()
            print(f"{mode_name:12s} {bucket:>3s}: loading (costs={not args.no_costs}) ...", flush=True)
            plan = cp.MLComputePlan.load_from_path(
                str(COREMLC / f"kokoro_decoder_har_post_{bucket}.mlmodelc"), mode
            )
            ops = plan.model_structure.program.functions["main"].block.operations
            rec = scan(plan, ops, with_costs=not args.no_costs)
            result["modes"][mode_name][bucket] = rec
            print(
                f"{mode_name:12s} {bucket:>3s}: {rec['ops_total']:5d} ops, "
                f"device_ops={rec['device_ops']}, "
                f"weight={rec['est_weight_by_device']} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    if OUT.exists():
        prev = json.loads(OUT.read_text())
        for m, bs in prev.get("modes", {}).items():
            dst = result["modes"].setdefault(m, {})
            for b, rec in bs.items():
                dst.setdefault(b, rec)
    result["diff"] = {}
    for bucket in BUCKETS:
        a = result["modes"].get("ALL", {}).get(bucket)
        g = result["modes"].get("CPU_AND_GPU", {}).get(bucket)
        if a is None or g is None:
            continue
        changed = {}
        for op in sorted(set(a["by_op"]) | set(g["by_op"])):
            da, dg = a["by_op"].get(op, {}), g["by_op"].get(op, {})
            if da != dg:
                changed[op] = {"ALL": da, "CPU_AND_GPU": dg}
        result["diff"][bucket] = {"ops_with_different_placement": changed}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
