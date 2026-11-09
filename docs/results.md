# Operator results

[中文](results.zh-CN.md) · [Contracts](contracts.md) · [Reproduce](reproduction.md)

The four formal suites share 150 workload IDs with Triton–Weft. Each selected
TileLang program ran through ordinary source JIT on the execution machine,
numerical checking, fresh-process cache restoration and native measurement.

![Native reference ratios and executed coverage](assets/performance.svg)

[Full operator CSV](assets/operators.csv) · [Summary](assets/summary.json) ·
[Original candidate and reference samples](assets/observations.json) ·
[Build and observation provenance](assets/provenance.json)

## Coverage and native performance

All 553 cases completed their target-native source JIT and numerical checks.

| Platform | JIT + numerical checks | Matched refs | Faster than ref | Median native speedup |
|---|---:|---:|---:|---:|
| SG2044 | 133/133 | 132 | 11 | 0.0580× |
| K1 | 137/137 | 136 | 9 | 0.1143× |
| K3 A100 | 136/136 | 135 | 10 | 0.1074× |
| x86 | 147/147 | 140 | 11 | 0.0375× |
| **Total** | **553/553** | **543** | **41** | — |

Speedup is **matched reference native time / TileLang native time**. Values above
one favor TileLang. The platform median gives each matched case one observation.
Dot leaves, complete projections and matrix tiles retain their own `scope` in
the CSV and detailed summary. The ten reference gaps retain numerical checks
and native timings with an empty external ratio.

| Reference gap | Platform cases |
|---|---:|
| `causal_attention`, one on each platform | 4 |
| `ime_centered`, `ime_centered_decode`, `ime_affine`, `ime_affine_decode` on x86 | 4 |
| `f16_projection_decode`, `f16_projection_prefill` with activation conversion on x86 | 2 |

These entries have no retained external observation matching the complete
contract. The separately registered x86 F16 variants preserving F32 activations
have their own matched comparisons.

## Reading the performance gaps

Normalization and activation quantization already benefit from the shared vector
compiler. For example, x86 LayerNorm reaches 3.80× and softmax 5.10×; SG2044
RMSNorm reaches 1.71× and LayerNorm 2.40× against their matched references.
Packed dot, complete projection and several serialized matrix programs
have substantial performance gaps. Native timings expose these gaps directly;
cold compilation and Python calls have separate columns.

The current author programs make several relevant implementation choices:

- [`packed.accumulate`](https://github.com/Tianchen-ai/tilelang-weft/blob/v0.1.15-weft.2/tilelang/weft/kernels/packed.py)
  uses typed integer partials for Q4_K/Q6_K and F64 scaled accumulation. Its
  general codebook path decodes 32 values, converts to F64 and reduces each segment.
- `packed.projection` quantizes activations inside the invocation, then traverses
  activation rows, output-column groups and packed records. Weight decoding is
  inside this tree, limiting reuse across activation rows.
- [`matrix.packed_x86`](https://github.com/Tianchen-ai/tilelang-weft/blob/v0.1.15-weft.2/tilelang/weft/kernels/matrix.py)
  constructs 64-element operands per quantization group. Matrix instructions
  accelerate the contraction; operand construction, packing and per-group
  updates remain part of the complete workload.

These are source and generated-program observations. Their individual timing
contributions require targeted optimization measurements. Retained artifacts and
source parameters make that work reproducible. Triton timings are an independent
comparison column; differences between author realizations remain explicit.

## Numerical coverage and execution evidence

Integers and serialized Encoding fields use exact checks. Floating-point results
retain each workload's existing tolerance. Projection decode checks every output
column; prefill checks the established 64-output sample and finiteness of every
output. Activation bytes follow the declared target rounding contract.

Every final row requires cold JIT, reuse of the compiled kernel, fresh-process
disk-cache restoration, Python hot-call timing, native samples and agreement with
the formal source binding. Source-equivalence audits identify observations
retained across documentation changes or unchanged Canonical programs; their
original package version remains recorded.

For the x86 compiler repair, all 147 retained Canonical programs were lowered by
both core versions. Four LLVM modules changed and those four cases were rerun;
the remaining 143 modules were identical. The latest core changes are confined
to x86 physical passes.

`used_extensions` records IME/AMX requirements from the artifact.
`selected_physical_engines`, architecture, VLEN and resource metadata describe its
lowering. Conditional runtime instruction observations have a separate field.

## Reference sources

| Platform | Selected native reference |
|---|---|
| SG2044 | Retained GGML CPU/RVV source and Clang 23 Release build |
| K1 | Retained GGML CPU/RVV build and its SpacemiT IME1 entries |
| K3 A100 | GGML/SpacemiT revision `6df9f75`, recorded RVV patches and IME2 entries |
| x86 | GGML revision `620d222`; oneDNN `3.13.4` for the two selected typed dense cases |

The shared engine's [reference catalog](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/src/weft_bench/catalogs/references.json)
defines source locations, function boundaries and preparation protocols. Each
baseline retains its original build and observation; this campaign reuses those
measurements under the matched contract. OpenBLAS remains an available registered
reference, while this formal selection uses the entries listed above.

`observations.json` publishes the original candidate, reference and matched
independent Triton samples,
shapes, input policies, numerical-check scope and native timing options. Private
execution paths are omitted. It accompanies the tables; the engine's local fixed
baseline indexes retain the full original comparison identity. Available source
bindings and producer versions identify the programs behind each sample set.

## Repairs included

Compiler repairs cover encoded field-write coordinates, masked local stores,
masked-load extents, unsigned RVV shifts, stale indexed-load planning and
axis-renamed Encoding subwindows. Q4_K/Q6_K typed partials retain their scale and
correction branches. An x86 short-halfword repair resolves an LLVM 23
instruction-selection crash in legal IQ1_S programs.

The shared Bench engine reuses immutable generated inputs across cold/cache
checks and native execution. It records the actual Python sampling budget and
resolves per-case native timing settings identically for individual and full-suite
runs. Fixed reference libraries and original numerical contracts are retained.

## Reproduce a row

Identify the program with `target`, `case`, `shape`, `scope`, `source_parameters`
and `backend_options`. `native_us` and `reference_us` determine the native ratio.
`cold_jit_ms`, `disk_cache_restore_ms`, `first_cache_hit_ms` and `python_hot_ms`
describe the separate user-facing JIT path.
The cold and fresh-process cache metrics include their first complete workload
call. They measure compile-and-launch or restore-and-launch, respectively;
the native/reference ratio uses only the native workload samples.
`compiler_api_ms` and `first_call_outside_compiler_ms` split the cold call using
the existing JIT events. Child-process CPU metrics and original compilation/cache
events accompany the observations, so a long workload call remains distinguishable
from compilation work.

```sh
tilelang-weft-bench show q4_k_projection_decode --target k1 --config .local/bench.local.json
tilelang-weft-bench op q4_k_projection_decode --target k1 --config .local/bench.local.json \
  --baseline output/baselines/k1 --output output/current/k1/q4_k_projection_decode
```

[Reproduction](reproduction.md) covers fixed references, complete suites,
explicit observation selection and regeneration of the table.
