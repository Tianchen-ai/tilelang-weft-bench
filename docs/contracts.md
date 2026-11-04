# Programs, storage and timing contracts

[中文](contracts.zh-CN.md) · [Handbook](index.md)

## One workload, explicit frontend bindings

The shared engine owns mathematical workloads, shapes, deterministic inputs,
numerical oracles and reference invocation. This repository's
`tilelang_weft_bench.catalogs.targets` files bind each workload to an installed
`tilelang.weft.kernels` module, factory symbol, source parameters, engine
constraint, grid and physical options. `show CASE` displays that binding.

| Work | TileLang source | Included work |
|---|---|---|
| Packed single/multi-row dot, 24 formats | `packed.vec_dot` | Block-local field decode, typed products, scale/min branches and accumulation |
| Packed projection prefill/decode | `packed.projection` | Activation quantization, caller-owned workspace, packed dot and output |
| F16 projection | `projection` | Declared F16 activation conversion, or explicitly preserved F32 activation |
| IME1/IME2 and AMX/VNNI tiles | `matrix`, `layouts` | Author-declared serialized layout and local contraction engine |
| Normalization, RoPE, softmax, attention, GEMM | `normalization`, `attention`, `linear` | Full registered workload and its declared preparations |

Encoding schemas live in `weft_core.formats.ggml`. The product's
`tilelang.weft.kernels.ggml_formats` exposes those schemas as TileLang Encoding
definitions. Persistent interleave declarations belong to `layouts`; numeric
decoding belongs to the ordinary author macros in `decoding`. Tables are explicit
input buffers. Bench supplies the same table bytes used by its independent oracle.

The packed suite decodes a bounded piece inside the admitted record's lifetime.
Source-declared finite-width partials and scale/correction branches remain part
of the program. Full dequantization to a persistent dense weight workspace is a
separate product composition with its own preparation protocol.

## Preparation and hardware

Projection activation quantization stays inside every measured invocation.
Prequantized dot/tile workloads receive already serialized inputs. Persistent
GGML model-default AMX packing follows the existing explicit preparation provider
and is recorded separately. Read `scope`, `persistent_inputs` and `preparation`
before comparing two records.

`engine="matrix"` constrains a particular contraction. Its selected local
instructions appear in the generated Physical IR and exported metadata. The
machine profile selects permitted hardware; it does not force every arithmetic
operation onto matrix hardware. An instruction in an artifact and observation of
a conditional execution branch are separate evidence fields.

The CSV's `used_extensions` and `required_extensions` describe the artifact's
additional extension contract, including IME or AMX. Base vector work is recorded
by `selected_physical_engines`, target architecture and VLEN; an empty additional
extension list does not imply scalar execution. `instruction_path=not_observed`
keeps conditional runtime instruction coverage explicit.

The exported compiler metadata supplies the selected engines, instruction
families and resource analysis. The CSV reads source parameters and physical
options from each retained JIT request and checks them against the formal
catalog binding. A case name containing `amx` or `ime` does not replace this
artifact evidence.

K3 A100 requires PID registration before CPU affinity. Its A100 group uses
VLEN1024; X100 uses VLEN256. The target probe checks the selected execution core.
One target machine runs one performance queue at a time.

## Numerical and measurement boundaries

Integer and serialized Encoding contracts are exact. Floating checks use the
shared workload's existing tolerances. Complete decode outputs and the existing
bounded prefill output sample retain their original check coverage; the result
records distinguish them.

| Metric | Boundary |
|---|---|
| Cold compile and launch | First ordinary TileLang JIT invocation |
| Disk-cache restoration | Fresh interpreter loading the same source specialization |
| First in-process cache hit | Second invocation in the initial interpreter |
| Python hot call | Cached factory invocation, binding, guards and native launch |
| Native workload | Exported typed C ABI call, including declared preparation |
| Native grid | Whole grid launcher, scheduling and synchronization |

The speedup against GGML or another native library uses the matching native
workload boundary. Python hot time is retained as a separate metric. The fixture
checks that hot calls reuse the compiled kernel. Compiler work, input generation,
numerical validation and output poisoning occur outside native timing.
Python hot sampling uses a time budget derived from the requested batch time;
long calls need fewer repeated samples. The CSV includes its actual sample count
and protocol. Native warmup, sample counts and reference matching stay fixed.
`disk_cache_restore_ms` measures the first JIT call in the fresh interpreter;
`first_cache_hit_ms` measures the second call in the initial interpreter. Input
preparation and Python process startup are outside both call windows.
