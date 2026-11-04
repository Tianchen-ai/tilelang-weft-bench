# TileLang–Weft Bench handbook

[中文](index.zh-CN.md) · [Source](https://github.com/Tianchen-ai/tilelang-weft-bench) · [Compiler](https://github.com/Tianchen-ai/tilelang-weft)

:::{toctree}
:hidden:

index.zh-CN
contracts
contracts.zh-CN
reproduction
reproduction.zh-CN
:::

This package selects native TileLang programs for the shared Weft Bench engine.
It covers 150 distinct operator IDs in 553 platform-specific cases. The compiler
package is `tilelang`; the Bench command is `tilelang-weft-bench`.

| Target | Cases | External references |
|---|---:|---:|
| SG2044 / RVV128 | 133 | 132 |
| K1 / RVV256 and IME1 | 137 | 136 |
| K3 A100 / RVV1024 and IME2 | 136 | 135 |
| x86 / SIMD, VNNI and AMX | 147 | 140 |
| Total | 553 | 543 |

## Install

Download the matching wheels from [Releases](https://github.com/Tianchen-ai/tilelang-weft-bench/releases).
Install CPU PyTorch for the execution host, then install the product and Bench:

```sh
python -m pip install --find-links ./wheels ./wheels/weft_core-*.whl \
  ./wheels/tilelang-*.whl ./wheels/triton_weft_bench-*.whl \
  ./wheels/tilelang_weft_bench-*.whl
tilelang-weft-bench list --kind targets
```

`triton-weft-bench` is the shared measurement engine's distribution name. The
TileLang workflow imports `weft_bench` and `tilelang_weft_bench`; its compiler is
the installed TileLang–Weft package and shared `weft-core` SDK.

Each execution host needs its own matching native wheels. RISC-V Python ABIs,
libc and x86 CPU features follow the individual wheel's release description.
The core SDK provides Clang 23, its resource headers and OpenMP support.

## Find and run an operator

Copy [the public configuration template](https://github.com/Tianchen-ai/tilelang-weft-bench/blob/main/configs/local.example.json)
to `.local/bench.local.json` and set the actual CPUs, libraries and installed SDK
paths. For SSH execution, set `deployment.host`, `deployment.workspace` and
`implementations.tilelang.python` to paths on that host.

```sh
tilelang-weft-bench list --kind kernels --target k3-a100 --config .local/bench.local.json
tilelang-weft-bench show q4_k_dot --target k1 --config .local/bench.local.json
tilelang-weft-bench op q4_k_dot --target k1 --config .local/bench.local.json \
  --output output/current/k1/q4_k_dot
```

The target process imports the installed author source and calls its ordinary
TileLang JIT factory. Compilation, native loading, cache restoration and execution
take place on that target. The subsequent native measurement uses the artifact
exported by that successful invocation, with the same input contract.

Read [contracts](contracts.md) to identify the author program and counted work,
then [reproduction](reproduction.md) for full suites, fixed references and result
interpretation.
