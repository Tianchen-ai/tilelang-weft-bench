# 算子结果

[English](results.md) · [测量合同](contracts.zh-CN.md) · [复现](reproduction.zh-CN.md)

四个正式套件与 Triton–Weft 共用 150 个 workload ID。每个选定的 TileLang
程序都在执行机通过普通源码 JIT 运行，完成数值检查、新进程缓存恢复和原生计时。

![原生参考倍率与执行覆盖](assets/performance.svg)

[完整算子 CSV](assets/operators.csv) · [汇总](assets/summary.json) ·
[候选与参考原始样本](assets/observations.json) ·
[构建与测量来源](assets/provenance.json) ·
[逐项最终代码比较](assets/compiler-emission.json)

## 覆盖范围与原生性能

553 项全部完成目标机普通源码 JIT 和数值检查。

| 平台 | JIT 与数值检查 | 匹配参考 | 快于参考 | 原生倍率中位数 |
|---|---:|---:|---:|---:|
| SG2044 | 133/133 | 132 | 106 | 1.3598× |
| K1 | 137/137 | 136 | 104 | 1.3176× |
| K3 A100 | 136/136 | 135 | 128 | 1.6569× |
| x86 | 147/147 | 140 | 130 | 1.2917× |
| **合计** | **553/553** | **543** | **468** | — |

倍率定义为 **匹配参考原生耗时 / TileLang 原生耗时**，大于 1 表示 TileLang 更快。
平台中位数对每个匹配用例取一个观测。单个 dot、完整 projection 和矩阵 tile
各自的计算范围保留在 CSV 的 `scope` 和详细汇总中。10 项参考缺口保留数值检查与
原生耗时，外部参考倍率留空。

| 参考缺口 | 平台用例数 |
|---|---:|
| 每个平台各一项 `causal_attention` | 4 |
| x86 的 `ime_centered`、`ime_centered_decode`、`ime_affine`、`ime_affine_decode` | 4 |
| x86 包含 activation 转换的 `f16_projection_decode`、`f16_projection_prefill` | 2 |

这些入口没有与完整合同匹配的已保留外部观测。x86 另外登记的保留 F32 activation
的 F16 变体有各自匹配的参考比较。

## Triton 性能继承

![TileLang 与 Triton 的配对原生耗时](assets/parity.svg)

选定的 553 项均有匹配的 Triton 对照，每项 TileLang 原生耗时不超过对应
Triton 的 **1.03 倍**。候选与对照使用同一批输入交替测量；候选就是已通过
TileLang 源码 JIT 数值检查后导出的同一份原生库。

K3 的两项 Q1 对照使用保留的 Triton Canonical 程序，经修复精确 I8 归约的
共享编译器重新生成。旧对照在对 `-128` 取负时溢出：边界输入得到 `-2048`，
正确结果应为 `0`；两种前端修复后的产物都通过该输入。
`triton_historical_us` 保留旧耗时，`triton_control_note` 标出修正的比较。

| 平台 | 匹配原生配对 | 3% 范围内 |
|---|---:|---:|
| SG2044 | 133 | 133/133 |
| K1 | 137 | 137/137 |
| K3 A100 | 136 | 136/136 |
| x86 | 147 | 147/147 |
| 合计 | 553 | 553/553 |

`native_us` 与 `triton_native_us` 来自这些同场配对。外部参考保留固定库与
原观测身份，其倍率使用 `reference_us / reference_candidate_us`。
冷 JIT、缓存恢复和 Python 调用各自单列，均在原生比较窗口之外。

## 作者程序与共享编译

TileLang 原生 TIRx 导入 Triton–Weft 同一套 Canonical 编译器。完整 packed workload
保留输入准备、Encoding、整数 partial、scale/min 支路和输出更新；正式绑定逐项
列明作者程序、分块、准备协议及目标物理参数。

- 短 record 与 K/IQ 格式使用有界的 entry/payload 或 plane Level。
  I16/I32 partial 在声明的浮点更新前完成；源码参数明确 record 窗口、行 cohort 和复用范围。
- 完整 projection 将 activation 量化保留在调用内。显式权重与 activation stage
  在作者指定的行列 cohort 中复用编码 panel；持久派生布局有单独声明的准备协议。
- IME1、IME2、VNNI 和 AMX 变体在作者程序中表达操作数布局与局部 contraction 意图。
  共享物理 passes 选择指令、局部 packing、转换和资源处理。
- 动态 RoPE 遍历保留运行时长度，并在写输出前取得两个输入快照。
  Indexed 程序保留明确的 lookup 位宽与 record metadata 读取，Encoding 提供真实存储几何。

前端修复保留标量 SSA、helper 轴、恒等 slice、字段投影和循环 carry。
共享编译器补齐 scalar tuple pack、Boolean 比较布局、整数归约、alias 与 capture
活跃性分析。成组 dense-memory 传输在真实读写前规范化选定的 carrier；矩阵结果接
逐元素运算时，即使操作数向量等宽，也按有序逻辑轴保留正确的 lane 坐标。
编译器依据这些 typed program 降低，不按 kernel 或量化格式名称分派实现。
每个候选均记录实际使用的 Clang 23 配置。

共享 widening 归约保留作者声明的整数 partial。零种子作为只读 RVV intrinsic 值
供给局部汇编，让 LLVM 在循环中复用它，归约顺序保持不变。SG2044 的 Q6_K dot
同场配对为 **TileLang 1.060 µs / Triton 1.054 µs**；生成代码减少了重复的
种子初始化与向量状态设置。

对于 indexed entry load，归约选择同时检查完整活跃集和有界的类型化 producer 链。
独立归约能够放入资源预算时，保留 intrinsic，让 LLVM 重叠查表供给与计算。
该选择消费已有物理操作、轴和生命周期。

K1 的 Q2_K prefill 选定作者程序将 U2 code 与 U4 scale 合入调用内的 I8 panel，
其精确范围为 `[0, 45]`。矩阵 contraction 跨 activation 行复用这个 panel；
I16 block-sum 修正仍采用向量路线。完整原生交替配对为 **TileLang 976.196 ms /
Triton 1446.994 ms**，相对保留的 Triton 程序为 **1.482×**。
activation 量化和 panel 填充均计入调用。复用结构来自作者程序；共同编译器保留
Encoding、矩阵布局、数值支路和正确的 lane 坐标。

## 数值范围与执行证据

整数和序列化 Encoding 字段按精确合同检查；浮点结果沿用 workload 已有容差。
projection decode 检查每个输出列；prefill 使用已有的 64 输出抽样，同时检查全部
输出的有限性。activation 字节遵循声明的目标舍入合同。

最终每行都要求冷 JIT 成功、复用同一编译 kernel、新进程磁盘缓存恢复、Python
热调用耗时、原生样本，以及与正式源码绑定一致。源码等价核对明确标识文档修改或
Canonical 程序未变时继承的测量；记录保留原包版本。

最终编译器重新导入全部 553 项选定的原始 TIRx 程序：357 项生成代码完全一致，
196 项仅增加只读 `restrict` 限定。实际执行体改变的程序已通过安装包验证和原生
配对后再选入结果；代码未变的程序保留原测量。

Canonical 与 Physical 核对定位编译器或作者程序改动影响的程序，再以目标机安装包
定向验证相关路径。未变程序保留原观测与包身份；来源文件记录源码绑定和共享 core 的提交。

`used_extensions` 来自产物的 IME/AMX 要求；`selected_physical_engines`、
架构、VLEN 和资源 metadata 展示 lowering。条件指令路径的运行观察另列。

## 参考来源

| 平台 | 选定的原生参考 |
|---|---|
| SG2044 | 已保留的 GGML CPU/RVV 源码与 Clang 23 Release 构建 |
| K1 | 已保留的 GGML CPU/RVV 构建及其 SpacemiT IME1 入口 |
| K3 A100 | GGML/SpacemiT `6df9f75`、已登记的 RVV 补丁与 IME2 入口 |
| x86 | GGML `620d222`；两个选定的 typed dense 用例使用 oneDNN `3.13.4` |

共同引擎的[参考目录](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/src/weft_bench/catalogs/references.json)
定义源码、调用边界与准备协议。各 baseline 保留原构建和原观测，本轮按匹配合同复用。
OpenBLAS 仍是已登记的可选参考；本次正式选择使用上表中的入口。

`observations.json` 公开候选、参考及匹配的独立 Triton 对照的原始样本、shape、输入策略、数值检查范围和
原生计时参数，移除机器私有路径，供表格核对使用。引擎的本地固定 baseline 索引
继续保留完整的原始比较身份。文件保留可取得的源码绑定和 producer 版本，便于定位实际程序。

## 本轮修复

修复覆盖 Encoding 字段写入坐标、带 mask 的局部更新、masked load 有效范围、
RVV 无符号移位、失效的 indexed-load 决定，以及逻辑轴重命名后的 Encoding 子窗口。
Q4_K/Q6_K typed partial 保留缩放与修正分支。x86 短 halfword 修复解决了合法
IQ1_S 程序触发的 LLVM 23 指令选择崩溃。

共同 Bench 引擎跨冷 JIT、缓存检查和原生执行复用不可变输入，记录 Python
实际采样预算，并对单项和完整套件使用相同的逐 case 原生计时配置。固定参考库和
原数值合同保持原有身份。

## 复现一项结果

通过 `target`、`case`、`shape`、`scope`、`source_parameters` 和
`backend_options` 确定程序。两种原生比较分别保留各自的样本：

- `native_us / triton_native_us` 是同场交替调用得到的 TileLang/Triton
  性能继承比例，每项不超过 1.03 才达标。
- `reference_us / reference_candidate_us` 是外部参考倍率。
  `reference_candidate_us` 来自源码 JIT 运行后导出的原生 workload，
  `reference_match` 标明与之匹配的固定参考。

`observations.json` 保留全部样本，不把两场比较的耗时混成第三个倍率。
`cold_jit_ms`、`disk_cache_restore_ms`、`first_cache_hit_ms` 和
`python_hot_ms` 分别描述正常用户 JIT 路径。
冷调用和新进程缓存指标都包含第一次完整 workload 调用，分别对应编译并启动、
恢复并启动。原生参考倍率仅使用原生 workload 样本。
`compiler_api_ms` 和 `first_call_outside_compiler_ms` 按已有 JIT 事件拆分冷调用；
同时公开编译子进程 CPU 耗时和原始编译、缓存事件，便于区分计算耗时与编译工作。

```sh
tilelang-weft-bench show q4_k_projection_decode --target k1 --config .local/bench.local.json
tilelang-weft-bench op q4_k_projection_decode --target k1 --config .local/bench.local.json \
  --baseline output/baselines/k1 --output output/current/k1/q4_k_projection_decode
```

[复现说明](reproduction.zh-CN.md) 包含固定参考、完整套件、显式结果选择和全量表生成命令。
