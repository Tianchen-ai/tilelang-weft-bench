# TileLang–Weft Bench

![源码 JIT 与匹配原生测量](docs/assets/workflow.svg)

[English](README.md) · [TileLang–Weft](https://github.com/Tianchen-ai/tilelang-weft)

本项目使用共同 Weft Bench 引擎，对原生 TileLang 作者程序进行算子比较。
四平台套件共553项：SG2044 133、K1 137、K3 A100 136、x86 147，
其中543项有匹配外部参考。K3的完整套件包含三项IME2。

算法与编译器属于产品；本仓库维护TileLang源码绑定、比较选择和结果入口。
输入、数值检查、原生计时、SSH与结果保存复用 `weft_bench`，无需安装Triton编译器。

```sh
python -m pip install --find-links ./wheels ./wheels/tilelang_weft_bench-*.whl
tilelang-weft-bench list --kind targets
tilelang-weft-bench list --kind kernels --target k3-a100
tilelang-weft-bench show q4_k_dot --target x86-1t --config bench.local.json
tilelang-weft-bench op q4_k_dot --target x86-1t --config bench.local.json \
  --output output/current/x86-1t/q4_k_dot
```

各执行机安装匹配的TileLang–Weft、`weft-core`与Bench包。
机器配置保存核心、工具链、参考库和SSH位置，不进入Git。
完整用例通过目标机JIT执行，编译、Python热调用与原生计算分别记录。

性能结果：**测试中 / Under evaluation**。
