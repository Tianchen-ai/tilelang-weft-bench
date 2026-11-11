# TileLang–Weft Bench 工具书

[English](index.md) · [源码](https://github.com/Tianchen-ai/tilelang-weft-bench) · [编译器](https://github.com/Tianchen-ai/tilelang-weft)

[算子结果](results.zh-CN.md) 展示原生倍率、JIT 指标、数值范围，以及性能差距对应的作者程序结构。

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

下载配套编译器和 Bench 资产。Pip 从目录中选择匹配当前 Python ABI 与宿主架构的
wheel：

```sh
gh release download weft-parity --repo Tianchen-ai/tilelang-weft \
  --pattern '*.whl' --dir wheels
gh release download weft-parity --repo Tianchen-ai/tilelang-weft-bench \
  --pattern '*.whl' --pattern constraints.txt --dir wheels
python -m pip install --find-links ./wheels -c ./wheels/constraints.txt \
  tilelang tilelang-weft-bench
python -m pip install --force-reinstall --no-deps --no-index --find-links ./wheels \
  -c ./wheels/constraints.txt tilelang weft-core triton-weft-bench tilelang-weft-bench
tilelang-weft-bench list --kind targets
```

执行上述命令前，先为主机安装 CPU PyTorch。x86 干净虚拟环境可用
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`；
RISC-V 使用对应平台的 CPU PyTorch 包。也可以直接从 GitHub Release 页面逐个下载
资产。

`triton-weft-bench` 是共用测量引擎的发行名称。TileLang 路径导入 `weft_bench` 与
`tilelang_weft_bench`，使用已安装的 TileLang–Weft 和共享 `weft-core` SDK 编译。

Release 的 `constraints.txt` 固定 TileLang `0.1.15+weft.2`、core `0.1.1`、共用
引擎 `1.0.1` 与本包 `0.1.0`。可用
`pip install -c constraints.txt --find-links ./wheels tilelang tilelang-weft-bench`
安装配套版本。

每台执行主机安装对应原生 wheel。RISC-V Python ABI、libc 和 x86 指令要求见各 wheel
的发行说明。core SDK 提供 Clang 23、resource headers 和 OpenMP 支持。

配套 core wheel 覆盖 x86-64 CPython 3.10，以及 RISC-V LP64D CPython
3.11/3.12/3.14。随包原生库要求 x86-64 glibc 2.34+、RISC-V glibc 2.38+，
以及 GLIBCXX 3.4.30+。TileLang wheel 使用对应宿主架构的 CPython 稳定 ABI；
core wheel 匹配具体解释器。

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
