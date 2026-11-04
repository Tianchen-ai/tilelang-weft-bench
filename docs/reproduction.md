# Reproduce and read comparisons

[中文](reproduction.zh-CN.md) · [Handbook](index.md)

Inspect the source and reference before execution:

```sh
tilelang-weft-bench show q4_k_projection_decode --target k1 --config .local/bench.local.json
tilelang-weft-bench list --kind baselines --target k1 --config .local/bench.local.json
tilelang-weft-bench list --kind suites --target k3-a100 --config .local/bench.local.json
```

For a focused development check, select the affected case. A complete suite uses
`full`; K3 `full` includes all 136 cases and its `ime2` subset contains the three
matrix cases.

```sh
tilelang-weft-bench op q4_k_dot --target k1 --config .local/bench.local.json \
  --check-only --output output/current/k1/q4_k_dot
tilelang-weft-bench op --suite full --target k3-a100 --config .local/bench.local.json \
  --baseline output/baselines/k3-a100 --output output/current/k3-a100/full
```

A fixed reference is a versioned library and a matching measurement contract.
Baseline indexes retain the original observations. When confirmed environmental
drift requires pairing, execute only the affected reference through the same
engine and keep its historical observation. Timing flags must match the selected
reference record; existing K3 cohorts include different warmup counts.

Reuse successful observations with the existing comparison command:

```sh
tilelang-weft-bench compare output/baselines/k1 output/current/k1/full \
  --baseline-impl ggml --output output/reports/k1
tilelang-weft-bench explain output/current/k1/full --case q4_k_dot
tilelang-weft-bench replay output/current/k1/full --case q4_k_dot \
  --output output/current/k1/q4_k_dot
```

`results.json` preserves samples, checks, compilation/cache observations and
artifact locations. `results.csv` is the engine's compact view. The published
full operator CSV adds source variants, native/reference timing, the independent
Triton comparison and JIT metrics. Use the workload's numerical-check record to
identify complete versus sampled checking.

A ratio greater than one means the TileLang native workload is faster. The ten
cases without a matched external reference retain a blank ratio and their own
correctness/timing result. Different source algorithms or preparation boundaries
are identified before interpreting a difference from Triton.

Results and artifacts live under this repository's ignored `output/` directory.
Public release tables and figures are sanitized copies; SSH hosts, machine paths,
models and credentials remain local. Re-running a target/case replaces its current
state through the engine's output transaction, retaining a failure as a failure.
