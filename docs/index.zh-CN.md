# TileLang–Weft Bench 工具书

[English](index.md) · [源码](https://github.com/Tianchen-ai/tilelang-weft-bench) · [编译器](https://github.com/Tianchen-ai/tilelang-weft)

本包为共用 Weft Bench 引擎选择原生 TileLang 作者程序，包含 150 个不同算子 ID、
553 个平台用例。编译器发行包为 `tilelang`，测试命令为 `tilelang-weft-bench`。

| 目标 | 用例 | 外部参考 |
|---|---:|---:|
| SG2044 / RVV128 | 133 | 132 |
| K1 / RVV256 与 IME1 | 137 | 136 |
| K3 A100 / RVV1024 与 IME2 | 136 | 135 |
| x86 / SIMD、VNNI 与 AMX | 147 | 140 |
| 总计 | 553 | 543 |

## 安装

从 [Releases](https://github.com/Tianchen-ai/tilelang-weft-bench/releases) 下载适配的 wheel。
先为执行主机安装 CPU PyTorch，再安装编译器与 Bench：

```sh
python -m pip install --find-links ./wheels ./wheels/weft_core-*.whl \
  ./wheels/tilelang-*.whl ./wheels/triton_weft_bench-*.whl \
  ./wheels/tilelang_weft_bench-*.whl
tilelang-weft-bench list --kind targets
```

`triton-weft-bench` 是共用测量引擎的发行名称。TileLang 路径导入 `weft_bench` 与
`tilelang_weft_bench`，使用已安装的 TileLang–Weft 和共享 `weft-core` SDK 编译。

Release 的 `constraints.txt` 固定 TileLang `0.1.15+weft.2`、core `0.1.1`、共用
引擎 `1.0.1` 与本包 `0.1.0`。可用
`pip install -c constraints.txt --find-links ./wheels tilelang tilelang-weft-bench`
安装配套版本。

每台执行主机安装对应原生 wheel。RISC-V Python ABI、libc 和 x86 指令要求见各 wheel
的发行说明。core SDK 提供 Clang 23、resource headers 和 OpenMP 支持。

## 查找并运行算子

将[公开配置模板](https://github.com/Tianchen-ai/tilelang-weft-bench/blob/main/configs/local.example.json)
复制到 `.local/bench.local.json`，填写实际 CPU、参考库和 SDK 路径。远端执行还需配置
`deployment.host`、`deployment.workspace`，以及远端的 `implementations.tilelang.python`。

```sh
tilelang-weft-bench list --kind kernels --target k3-a100 --config .local/bench.local.json
tilelang-weft-bench show q4_k_dot --target k1 --config .local/bench.local.json
tilelang-weft-bench op q4_k_dot --target k1 --config .local/bench.local.json \
  --output output/current/k1/q4_k_dot
```

目标进程从已安装包导入作者程序，调用普通 TileLang JIT factory。编译、原生加载、
缓存恢复和执行均在目标机完成。随后测量该次成功调用导出的原生产物，保持同一输入合同。

[合同说明](contracts.zh-CN.md)介绍作者程序、Encoding 和计时内的工作；
[复现指南](reproduction.zh-CN.md)介绍全量套件、固定参考及结果含义。
