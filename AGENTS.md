# TileLang–Weft Bench

- This repository owns TileLang candidate bindings, suite selections, comparison
  documentation and result presentation. Normal author kernels belong to
  `tilelang.weft.kernels`; compiler and runtime belong to the shared Weft core.
- Reuse `weft_bench` workloads, inputs, numerical checks, SSH, timers and output
  transactions. Do not introduce a second runner, oracle or hidden Triton source.
- Use the installed TileLang product on each execution target. Source JIT,
  exported native computation, grid and Python hot calls have separate timing
  boundaries. Preserve every case's preparation, dtype, shape and threading.
- The formal full set is SG2044 133, K1 137, K3 A100 136 and x86 147: 553 cases,
  543 with matched references. K3 full includes its three IME2 cases.
- Keep machine configuration and credentials in `.local/`. Results and artifacts
  use stable `output/` paths; fixed references are read-only indexes. Keep the
  latest state and necessary failure evidence, not dated tuning directories.
- Only claim correctness and performance supported by actual installed-package
  execution. Fix the layer responsible for a failure, retaining numerical
  tolerances. An unmatched reference remains unmatched.
- Ordinary development runs affected cases. Full suites run when requested.
  Keep English and Chinese documentation aligned and public data sanitized.
