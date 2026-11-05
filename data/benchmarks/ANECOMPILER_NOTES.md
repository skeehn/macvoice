# ANECompiler.framework internals — Phase 2 study (2026-11-05)

Ground truth for the v0.1.1 placement study (`compute_plan.json`, README
"Why ANE wins at 7–15s"). Remote-engineering record of the actual code the
wedged `ANECompilerService` daemon runs. Companion to the Phase 1 notes:
daemon XPC API, JIT broker, BSSLimit/shouldEnforceSizeLimits (evidence
ev_6bb8b12e / ev_008453ab / ev_54e9ffb6 / ev_d3ae0b82 / ev_107f32c0 / ev_bb833479).

## Target provenance

- Binary: ANECompiler.framework extracted from the macOS 26.0 (Tahoe)
  dyld shared cache via `ipsw dyld extract … ANECompiler --output .`
  → `/private/var/folders/sb/…/opencode/ane-extract/ANECompiler`
  47,020,696 bytes, sha256 ccd3cbfb…7c94e4 (as loaded, arm64).
- The on-disk framework is a masked stub (only _CodeSignature/Resources);
  the real code lives only in the cache. The daemon binary
  (sha256 593a0c98…742d5) links it as its compile backend.
- Method: Ghidra 12.1.4 headless import + full auto-analysis (425 s,
  87,676 functions, project ghidra-ane/ane-project). Decompiles in
  `ghidra-ane/out/` (out/fn_*, out/b2_*–b5_*), and deterministic
  disassembly extraction by **llvm-objdump + symbol-aware call/loop
  listing** (scripts inline in the session log). Analysis caveat:
  Ghidra decompiles the biggest ANECompiler functions
  (`_ANECCompile` 20KB, `ZinMirPrepareLayers` 9.5KB,
  `ZinIrPerf::CalculatePerfDescriptor`) as truncated stubs because the
  compiler emits PAC-indirect tail-call trampolines (the
  `_ZinComputeProgramGetProcedureNameFromGPUThread` helper, 539 call
  sites) that break flow reachability from the entry; the same bodies
  read fine as disassembly, so the structure below comes from
  call-graph extraction, which is complete.

## 1. Entry points (exported C ABI)

| symbol | address | notes |
|---|---|---|
| `_ANECCompile` | 0x2236b7c24 | ~20KB megabody; the whole per-procedure pipeline + serialize (below) |
| `_ANECCompileJIT` | 0x2249534f0 | separate JIT entry, big prologue |
| `_ANECCompileOnline` | 0x22362d504 | **stub: mov w0,#1; ret** (8 bytes) |
| `_ANECCompileOffline` | 0x22362d50c | **stub: mov w0,#1; ret** (8 bytes) |

The Online/Offline exports are inert in this build — all real compile
paths go through `_ANECCompile` / `_ANECCompileJIT`. (Observed via
llvm-objdump on exact symbols.)

`ANECPrepare` (0x2236b5804) dispatches on `_ANECGetCompilerFileFormat`:
1=ANECIR → `ANECCreatePrepareInfoFromANECIR`, 2=MIL file → `…FromMILFile`,
3=MLIR → `…FromMLIR`; everything returns `vector<ANECProcedureInfo>`.

## 2. The stage ladder (from `_ANECCompile` + `ZinCompilerCoreClassic::CompileProcedure` 0x223627600–0x223627f77)

Per procedure (`vector<ANECProcedureInfo>`), in first-use call order:

    ZinCompilerCoreClassic ctor(params, ctx, idents, nameMaps, dict, int, status)
    ZinIrCompilationStatus::AddNetwork
    ZinCompilerCore::PreProcessControlFlowGraph
    ValidateControlFlowGraph
    OptimizeControlFlowGraph
    RunMirPrepareIr
    RunFindInherentParallelism
    [GetCFGraph] RunMirBuilder(CFG)                 0x2237ea8ec  ~16KB body
    RunTaskScheduler(CFG)                           0x2237eaa34  ~35KB body
    ValidateMirInfo
    RunCPAllocator(CFG)                             (CP allocator + L2/DRAM banks)
    RunRegisterRelocationMapBuilerPreRegisterAllocator
    MirPrepareControlFlow / ZinMirSetZeroPad3Run
    RunRegisterSpilling
    RunRegisterRelocationMapBuilerPostRegisterAllocator
    ZinTileHeightErrata::Workaround
    RunMultiSegmentSpilling
    ZinMirEnableDoubleRateMode / ZinMirEnableAcceleratedTexturing
    RunHazardAnalysis
    RunRemoteDependencyAnalysis
    RunPieceGeneration
    RunMemCacheAllocation
    UpdateFinalKernelSHA
    SetBinaryPoint
    SetSplitRowCompute
    ZinIrContext::SetPrivateMemoryPools / ZinIrLocalRegAlloc::TransferDramPool
    DumpDebugProfilingInfo
    SetTDExecutionTime        ← per-TaskDescriptor execution time recorded here
    RunCachePrefetch
    RunContextSwitch          ← owner of the "Task %zd execution time (%lf ms)
                                 exceeds the specified context switch latency
                                 (%f ms)" check
    RunKernelBufferControl
    RunComputeAddressTranslationRegisters
    QualifyOnImbalanceRatio
    RunHandleMultiAneSynchronization
    RunMirLegalizeEmptyExitBasicBlock
    DumpLayerStats
    SetLiveIOAttributes        ← last stage; then linker/serialize:
    ZinComputeMutableProgramPatchRuntimeGraph / SerializeRuntimeGraph
    ZinAneLinker ctor/AddObject/Link → ZinIrConstManager::BuildMachO
    ZinObjectGenWriteSymbolData → HWX bytes; plist status out.

Adjacent big bodies in the same TU (candidates that scale with model size):

    ZinCompilerCoreClassic::BuildLayerGraph             0x223625bac ~14KB
    ZinCompilerCoreClassic::RunRegisterAllocator        0x2236261b4 ~14.5KB
    RunTaskScheduler                                    ~35KB (FM inlined above)

## 3. The prepare/fusion ladder (`ZinMirPrepareLayers`, 0x2247a1c9c–0x2247a4207, ~9.5KB, 125 sub-calls, 26+ cold paths)

In call order (first-use): `ZinIrOpt::ReverseCSE`
(TraverseForward lambda → graph rewrite), CollapseQuantDequant,
RemoveSingleChannelEwOpCopy, ZinMirLayerHoisting::Execute,
ScaledEW/WithConstIn GOC fusion, CollapseTranspose,
ZinMirPadOptimization, InputViewOpt, MergeDeconvConv, PreFusionCSE,
**ZinMirLayerFusion::Run (the fusion rounds)**, OptimizeToChannelLast,
ZinMirGOCEngineReassignment, Remove8BitToF16Copy,
ZinMirOptFullyConnectedLayer, … (125 unique targets; backward loop arcs
x9, x7, x6, x3, x2… — repeated multi-pass rewrites, each pass O(graph)).
This is the M4-article "ZinMirPrepareLayers two fusion rounds" confirmed
on M3/Tahoe.

## 4. Where the per-op cost numbers come from (what MLComputePlan consults)

- `ZinPerfUtil::GetTaskExecutionTimeEstimate` (0x224670990, ~1.5KB,
  fully decompiled): allocates `vector<ZinPerfUtil::TaskStats>` sized by
  `ZinIrDeviceMesh::GetNumAnes()`; then a **linear loop over every
  ZinIrOpLayer**: for non-NoOp ANE layers, calls
  `ZinANELayer::CalculatePerf(ZinIrPerf&, ZinPerfDescriptor&)`, reads
  `ZinBondedInfo::GetAneIndex()`, applies L2-lookup latency fixups
  (`ZinEngineLayerMirInfo::HasChainRead`,
  `ZinMirL2Config::GetL2SrcDep/HasAlias`), and emplaces TaskStats per
  (layer, ane). No nested tiling loop in the estimator itself.
- `ZinANELayer::CalculatePerf` (0x2236c4678) → forwards to the
  CustomPerfInfo overload (0x2236c46f8) →
  `ZinIrPerf::CalculatePerfDescriptor` (0x22467864c, 717B) which
  dispatches **`ZinPEPerf::CalculatePerf` / `ZinNEPerf::CalculatePerf`**
  (the actual per-op cycle-count models).
- Driver: `ZinPerfUtil::ZinDebugPrintPerfEstimateParams`
  (0x224671378): asserts **"Must run scheduler first"**, walks the CFG's
  schedule map (`map<ZinIrBasicBlock*, vector<ZinIrOpLayer*>,
  ScheduleComparator>`, stored in cfg+0x168 by
  `ZinIrControlFlowGraph::SetScheduleMap`), and calls the estimator once
  per basic block. The `_perf_all.csv` vocabulary
  (core_cycle_count, l2_cycle_count, ne_cycle_single/double_buffering,
  l2_bcast, lw_writeback, power(W), estimated_time(ms), tile_height,
  num_workunits, active_ne, sh_min/pref/max, ocg_size) is this family's
  dump schema (`ZinPerfUtil::WritePerfCsvHeader` right next to it).
- Tuning: `ZinIrCompilerParameters::setCostModelClusterThreshold`
  stores an int at params+0x2f4; `ZinMirGraphSplitCostModel::GetSplitCost`
  **base class returns a zero constant** (subclasses override —
  `ZinMirGraphSplitLatencyCostModel` is the real one).
- MLIR-side placement option family for the ANE-region formation pass
  (strings, from `mps-to-anec`): `use-cost-model`,
  `force-host-placement`, `force-ane-placement`, `print-estimated-cost`,
  `print-placement-report`, `ane-max-ops`, `ane-max-liveio`,
  `gpu-core-count`, target families `anec.A12…anec.A18` (M3 → A17,
  observed default A12 family string only — family binding is an
  UNRESOLVED observation).

## 5. Wedge assessment for the 30s bucket (macvoice ALL-mode plan load)

Observations:
- All decompiled estimator structures are linear in per-op-layer work;
  no detected unbounded scalar loop (backward arc counts per body are
  small: ANECCompile arcs x1–singles; PrepareLayers x9/x7/x6 multi-pass).
- The runtime-documented work budget is size-based only (Phase 1:
  BSSLimit/DRAMLimit, daemon `maxModelMemorySize` XPC input); nothing
  time-based anywhere in the backend (.strings: no timeout/deadline).
- Cost-heavy spots that scale super-linearly with tensor size are, by
  structure: `RunCPAllocator` (CpAllocGraph enumeration,
  `LayerCycleAndFootprintEstimator` per layer/ANE),
  `RunMemCacheAllocation`, the spatial-splitter cluster search
  ("Can't tile subgraph b/c SIP > budget", "Extracting cluster at High
  Pressure Region"), and per-ANE TaskStats accumulation under
  `GetTaskExecutionTimeEstimate` (per-ANE × per-BB × per-layer).

Inference (marked as such):
- The 30s plan-load wedge (daemon 100% CPU; client parked after ~1.5min)
  fits a combinatorial search (allocator/splitter) hitting a
  shape-space blow-up, since no external watchdog exists to time-bound
  it. Not yet PROVEN: which loop is the burning one requires live
  stepping (in-process `ANECCompile` + lldb, the maderix method) — a
  Phase 3 candidate.
- The 3s-vs-7s/15s asymmetry (ALL-mode overcommits ANE at 3s, rebalances
  by 7s) is consistent with the cost model being
  cycle-count/descriptor-based with fixed per-op overhead that the
  weight aggregation (`MLComputePlanCost.weight`) then normalizes —
  i.e., the runtime undersells fixed dispatch cost at small sizes.

Unknowns (explicit):
- Which exact loop burns in the 30s case (needs dynamic trace).
- Why Online/Offline variants exist as no-ops in this build (they
  return 1 and nothing in the observed paths routes through them).
- `anec.A17` family binding for M3 not observed (only option strings).

## Files

- Decompiles: `opencode/ghidra-ane/out/fn_*.c`, `out/b2_*`, `out/b3_*`,
  `out/b5_*` (the b5/megabody outputs are intentionally short — see
  caveat above; structure came from call-graph extraction).
- Disassembly dumps: `/tmp/anec_compile.asm` (4989 insns),
  `/tmp/compileproc.asm`, `/tmp/preparelayers.asm`,
  `/tmp/perfdesc.asm`.
- Mock of the ghidra project stays at `opencode/ghidra-ane/ane-project`
  (78MB) for further passes.
