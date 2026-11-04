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
reference record. The versioned catalog records the existing K3 warmup cohorts
and the 200 ms warmup required by K3 I4 affine and x86 AMX Q6_K prefill.
`show CASE` displays the resolved measurement options. Single-case and `full`
commands use them automatically; explicit timing flags override the corresponding
options.
The example configuration permits up to two hours for each build or execution
process. This budget includes source JIT and input preparation; it does not
change the timed sample window. Read-only generated inputs are captured once
and reused by cache verification and the native worker.

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

Generate the presentation table from explicitly selected runs and fixed references:

```sh
python -m pip install --find-links ./wheels -c ./wheels/constraints.txt \
  'tilelang-weft-bench[plots]'
python -m tilelang_weft_bench.report \
  --run sg2044=output/current/sg2044/selected --reference sg2044=output/baselines/sg2044 \
  --run k1=output/current/k1/selected --reference k1=output/baselines/k1 \
  --run k3-a100=output/current/k3-a100/selected --reference k3-a100=output/baselines/k3-a100 \
  --run x86-1t=output/current/x86-1t/selected --reference x86-1t=output/baselines/x86-1t \
  --triton x86-1t=output/comparisons/x86-1t --plot --output output/reports/latest
```

The plotting extra is needed on the report host. Execution hosts use the ordinary
Bench package.

Final reporting rejects incomplete JIT/cache/hot-call evidence, missing native
samples and unmatched required references. `--partial` retains incomplete rows
explicitly during development. Projection prefill uses the existing 64-output
sample and checks every output for finiteness; decode checks every column.
Quantized activation bytes are checked against the declared target storage rule.

Results and artifacts live under this repository's ignored `output/` directory.
Public release tables and figures are sanitized copies; SSH hosts, machine paths,
models and credentials remain local. Re-running a target/case replaces its current
state through the engine's output transaction, retaining a failure as a failure.
