# 程序、存储与计时合同

[English](contracts.md) · [工具书](index.zh-CN.md)

## 共用 workload，明确绑定前端

共用引擎维护数学 workload、shape、确定性输入、数值 oracle 和参考调用。
本仓库 `tilelang_weft_bench.catalogs.targets` 为每项绑定已安装的
`tilelang.weft.kernels` 模块、factory、源码参数、engine、grid 和物理选项。
`show CASE` 展示解析后的绑定。

| 工作 | TileLang 源码 | 计入的工作 |
|---|---|---|
| 24 格式单行／多行 packed dot | `packed.vec_dot` | block 内字段解码、typed partial、scale/min 支路与累加 |
| packed projection prefill/decode | `packed.projection` | activation 量化、调用方 workspace、packed dot 与输出 |
| F16 projection | `projection` | 声明的 F16 activation 转换，或明确保留 F32 activation |
| IME1/IME2、AMX/VNNI tile | `matrix`、`layouts` | 作者声明的序列化布局及逐 contraction engine |
| 归一化、RoPE、softmax、attention、GEMM | `normalization`、`attention`、`linear` | 登记的完整 workload 及准备工作 |

Encoding schema 定义在 `weft_core.formats.ggml`，产品中的
`tilelang.weft.kernels.ggml_formats` 将其暴露为 TileLang Encoding。
持久 interleave 声明位于 `layouts`，数值解码位于普通作者宏 `decoding`。
查表数据作为显式输入，Bench 提供与独立 oracle 相同的 table bytes。

packed 套件在已加载 record 的生命周期内解码有界片段。源码声明的有限位宽 partial、
scale 和 correction 支路保留在程序中。完整解码到持久 dense 权重 workspace 的组合
是产品中的另一种程序，遵循自己的准备协议。

## 准备工作与硬件

projection 的 activation 量化在每次测量调用内。预量化 dot/tile 接收已序列化输入。
GGML model-default AMX 的持久 packing 使用既有显式 provider，并独立记录。
比较前查看 `scope`、`persistent_inputs` 和 `preparation`。

`engine="matrix"` 约束具体 contraction。选中的局部指令写入 Physical IR 和导出
metadata。机器 profile 表达允许的能力；同一程序的其他计算仍可采用向量。
产物包含某条指令与运行时观察到条件分支是不同的证据字段。

CSV 的 `used_extensions`、`required_extensions` 表示产物的附加扩展合同，
包括 IME 或 AMX。基础向量计算另由 `selected_physical_engines`、架构与 VLEN
展示；附加扩展列表为空不能解释为标量执行。`instruction_path=not_observed`
明确保留条件指令路径尚未观察的状态。

导出的编译 metadata 提供所选 engine、指令类别和资源分析。CSV 从每项保留的
JIT request 读取源码参数与物理配置，并核对正式 catalog 绑定。用例名称中的
`amx` 或 `ime` 不能代替实际产物证据。

K3 A100 必须先注册 PID，再设置 affinity。A100 核组为 VLEN1024，X100 为 VLEN256。
目标探测检查实际执行核心。同一机器同时只运行一组性能队列。

## 数值与计时边界

整数和序列化 Encoding 保持精确合同；浮点沿用 workload 的既定容差。
decode 完整输出和 prefill 既有有界输出抽样保持原检查范围，并分别记录。

| 指标 | 边界 |
|---|---|
| 冷编译与调用 | 第一次普通 TileLang JIT 调用 |
| 磁盘缓存恢复 | 新解释器加载同一源码特化 |
| Python 热调用 | 缓存 factory 调用、参数绑定、guard 与原生启动 |
| 原生 workload | 导出的 typed C ABI 调用，包含声明的准备工作 |
| 原生 grid | 完整 grid launcher、调度与同步 |

对 GGML 等原生库的倍率采用匹配的原生 workload 边界，Python 热调用独立保留。
fixture 检查热调用复用同一编译 kernel。编译、输入生成、数值检查和输出污染在原生计时外。
