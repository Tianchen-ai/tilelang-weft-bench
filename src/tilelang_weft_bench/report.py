"""Present saved engine observations; no compilation, execution or timing."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path

from weft_bench.output import OutputRun
from weft_bench.results import apply_baselines, contracts_match, read

from .catalogs import target_catalog, target_names

FIELDS = (
    "target", "case", "shape", "scope", "status", "correctness", "native_us",
    "reference", "reference_us", "reference_candidate_us", "reference_speedup", "reference_match",
    "triton_native_us", "triton_match", "source_jit_native_us", "triton_historical_us",
    "tilelang_over_triton", "triton_gap_percent", "parity_status",
    "cold_jit_ms", "compiler_api_ms", "first_call_outside_compiler_ms",
    "cold_child_cpu_ms", "cache_hit_child_cpu_ms", "first_cache_hit_ms", "disk_cache_restore_ms",
    "python_hot_ms", "python_hot_samples", "python_hot_protocol", "disk_cache_hit", "same_compiled_kernel", "native_entry",
    "source", "symbol", "source_parameters", "backend_options", "binding_match", "used_extensions",
    "required_extensions", "selected_physical_engines", "target_arch", "target_vlen_bits",
    "instruction_path", "instruction_families", "resource_analysis", "runtime_cpus", "tilelang_version",
    "warmup", "warmup_ms", "batch_ms", "sample_count", "data_cache", "sample_order", "native_protocol", "numerical_check_scope",
)


def mapping(values):
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        if name not in target_names() or name in result:
            raise ValueError("each input must identify one distinct catalog target")
        result[name] = path
    return result


def observation_index(path, implementation="tilelang"):
    selected = {}
    for row in read(path):
        if row.implementation != implementation:
            continue
        if row.case in selected:
            raise ValueError("combine explicitly selected observations with checkpoint before reporting duplicates")
        selected[row.case] = row
    return selected


def artifact_metadata(row):
    location = row.details.get("artifacts", {}).get("weft")
    if not location:
        return {}
    filename = Path(location) / "kernel.native.json"
    if not filename.is_file():
        raise FileNotFoundError(filename)
    return json.loads(filename.read_text())


def source_request(row):
    location = row.details.get("jit", {}).get("diagnostics")
    if not location:
        raise ValueError(f"missing source JIT diagnostics: {row.target}/{row.case}")
    return json.loads((Path(location) / "jit-request.json").read_text())


def physical_evidence(metadata):
    abi = metadata.get("kernel_abi", {})
    compilation = metadata.get("compilation", {})
    operations = abi.get("physical_operations", {}).get("operations", [])
    contractions = compilation.get("contractions", [])
    engines = {operation["engine"] for operation in operations if operation.get("engine")}
    engines.update(operation["selected_engine"] for operation in contractions if operation.get("selected_engine"))
    families = {operation["instruction_family"] for operation in contractions if operation.get("instruction_family")}
    resources = {key: value for key, value in (
        ("physical_resources", abi.get("physical_resources")),
        ("target_resources", compilation.get("target_resources")),
        ("local_storage", compilation.get("local_storage")),
        ("scope", compilation.get("resource_estimate_scope")),
    ) if value is not None}
    return sorted(engines), sorted(families), resources


def require_jit_evidence(row, runtime, restored):
    label = f"{row.target}/{row.case}"
    checked = ("within-tolerance", "bit-exact", "sampled-64-exact")
    if not any(marker in row.correctness for marker in checked):
        raise ValueError(f"missing native numerical check: {label}")
    if runtime.get("status") != "ok" or not any(
        marker in runtime.get("correctness", "") for marker in checked
    ):
        raise ValueError(f"missing source JIT numerical check: {label}")
    cache = restored.get("compilations", [])
    if runtime.get("same_compiled_kernel") is not True or not cache or not all(item["cache_hit"] for item in cache):
        raise ValueError(f"missing source JIT/cache evidence: {label}")
    restore_ms = restored.get("cold_compile_and_launch_ms")
    if restored.get("status") != "ok" or restore_ms is None or not math.isfinite(restore_ms) or restore_ms <= 0:
        raise ValueError(f"missing fresh-process disk-cache restoration: {label}")
    for metric in ("cold_compile_and_launch_ms", "first_cache_hit_launch_ms"):
        value = runtime.get(metric)
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(f"missing {metric}: {label}")


def native_pairs(path):
    selected = {}
    for row in read(path):
        if row.implementation not in ("tilelang_native", "triton_control"):
            raise ValueError("native pairs require tilelang_native and triton_control observations")
        pair = selected.setdefault(row.case, {})
        if row.implementation in pair:
            raise ValueError(f"duplicate native pair observation: {row.target}/{row.case}")
        pair[row.implementation] = row
    return selected


def checked_pair(source, pair):
    label = f"{source.target}/{source.case}"
    if set(pair) != {"tilelang_native", "triton_control"}:
        raise ValueError(f"incomplete native pair: {label}")
    candidate, control = pair["tilelang_native"], pair["triton_control"]
    if not contracts_match(candidate, control):
        raise ValueError(f"native pair has different measurement contracts: {label}")
    # The source run proves JIT and numerical behavior; only the alternating
    # pair supplies comparison timings. Match the source workload with that
    # one intentional order change, retaining every other measurement field.
    source_measurement = dict(source.details.get("measurement", {}))
    if source_measurement.get("sample_order", "serial") not in ("serial", "alternating"):
        raise ValueError(f"unknown source measurement order: {label}")
    source_measurement["sample_order"] = "alternating"
    paired_source = replace(source, details={**source.details, "measurement": source_measurement})
    for row, producer in ((candidate, "tilelang"), (control, "triton")):
        if row.status != "ok" or not contracts_match(paired_source, row):
            raise ValueError(f"native pair does not match the checked source workload: {label}")
        if not any(marker in row.correctness for marker in ("within-tolerance", "bit-exact", "sampled-64-exact")):
            raise ValueError(f"native pair lacks a numerical check: {label}")
        if row.details.get("producer") != producer:
            raise ValueError(f"native pair has an incorrect compiler producer: {label}")
        if not row.samples_ms or row.median_ms is None or not math.isfinite(row.median_ms) or row.median_ms <= 0:
            raise ValueError(f"native pair has no finite latency samples: {label}")
        if row.details.get("measurement", {}).get("sample_order") != "alternating":
            raise ValueError(f"native pair was not measured in alternating order: {label}")
    source_artifact = Path(source.details["artifacts"]["weft"])
    paired_artifact = Path(candidate.details["artifacts"]["weft"])
    if source.details.get("entry") != candidate.details.get("entry") or source.details.get("launch", {}) != candidate.details.get("launch", {}):
        raise ValueError(f"native pair changes the checked source launch: {label}")
    for filename in ("kernel.weftir", "kernel.llir" if (source_artifact / "kernel.llir").is_file() else "kernel.c"):
        if (source_artifact / filename).read_bytes() != (paired_artifact / filename).read_bytes():
            raise ValueError(f"native pair is not the source-JIT artifact: {label}: {filename}")
    return candidate, control


def records(runs, references, triton, *, partial, paired=None):
    output = []
    for target in target_names():
        catalog = target_catalog(target)
        current = observation_index(runs[target]) if target in runs else {}
        baseline = read(references[target]) if target in references else []
        old = read(triton[target]) if target in triton else []
        pairs = native_pairs(paired[target]) if paired and target in paired else {}
        unknown = set(current) - set(catalog["cases"])
        if unknown:
            raise ValueError(f"observations outside {target}'s formal suite: {sorted(unknown)}")
        for name, selected in catalog["cases"].items():
            candidate = selected["candidate"]
            record = dict.fromkeys(FIELDS, "")
            record.update(target=target, case=name, shape=json.dumps(selected["shape"], sort_keys=True),
                          scope=selected["scope"], status="not_executed", reference=selected["reference"] or "",
                          reference_match="unmatched" if selected["reference"] else "catalog_gap",
                          source=candidate["source"], symbol=candidate["symbol"],
                          source_parameters=json.dumps(candidate.get("constexprs", {}), sort_keys=True),
                          backend_options=json.dumps(candidate.get("backend_options", {}), sort_keys=True))
            row = current.get(name)
            if row is None:
                if not partial:
                    raise ValueError(f"missing execution: {target}/{name}")
                output.append(record)
                continue
            if row.shape != selected["shape"]:
                raise ValueError(f"nonformal shape: {target}/{name}: {row.shape}")
            record.update(status=row.status, correctness=row.correctness, scope=row.scope,
                          native_us=row.median_ms * 1000 if row.median_ms is not None else "",
                          source_jit_native_us=row.median_ms * 1000 if row.median_ms is not None else "",
                          parity_status="not_measured")
            if row.status != "ok" and not partial:
                raise ValueError(f"unresolved execution: {target}/{name}: {row.status}")
            if selected["reference"]:
                apply_baselines([row], baseline, selected["reference"])
                record["reference_match"] = "matched" if row.speedup is not None else "unmatched"
                if row.speedup is not None:
                    record.update(reference_us=row.details["baseline_median_ms"] * 1000,
                                  reference_candidate_us=row.median_ms * 1000,
                                  reference_speedup=row.speedup)
                elif not partial:
                    raise ValueError(f"unmatched required reference: {target}/{name}: {row.details.get('comparison_note')}")
            else:
                record["reference_match"] = "catalog_gap"
            matches = [item for item in old if item.implementation in ("weft", "triton_control", "triton_source")
                       and item.status == "ok"
                       and item.details.get("producer") == "triton"
                       and item.median_ms is not None and contracts_match(row, item)]
            record["triton_match"] = "historical_only" if len(matches) == 1 else "unavailable" if not matches else "ambiguous"
            if len(matches) == 1:
                record["triton_historical_us"] = matches[0].median_ms * 1000
            if name in pairs and row.status == "ok":
                paired_candidate, control = checked_pair(row, pairs[name])
                ratio = paired_candidate.median_ms / control.median_ms
                record.update(native_us=paired_candidate.median_ms * 1000,
                              triton_native_us=control.median_ms * 1000,
                              triton_match="paired", tilelang_over_triton=ratio,
                              triton_gap_percent=(ratio - 1) * 100,
                              parity_status="passed" if ratio <= 1.03 else "over_3_percent")
            if not partial and record["parity_status"] != "passed":
                raise ValueError(f"unresolved Triton performance inheritance: {target}/{name}: {record['parity_status']}")
            jit = row.details.get("jit", {})
            runtime = jit.get("runtime", {})
            cache = jit.get("disk_cache", {}).get("compilations", [])
            timed_row = pairs[name]["tilelang_native"] if record["triton_match"] == "paired" else row
            measure = timed_row.details.get("measurement", {})
            metadata = artifact_metadata(row) if row.status == "ok" else {}
            if row.status == "ok":
                request = source_request(row)
                binding_match = (
                    "module:" + request["source_module"] == candidate["source"]
                    and request["symbol"] == candidate["symbol"]
                    and request["constexprs"] == candidate.get("constexprs", {})
                    and request["backend_options"] == candidate.get("backend_options", {})
                )
                record.update(source="module:" + request["source_module"], symbol=request["symbol"],
                              source_parameters=json.dumps(request["constexprs"], sort_keys=True),
                              backend_options=json.dumps(request["backend_options"], sort_keys=True),
                              binding_match=binding_match)
                if not binding_match and not partial:
                    raise ValueError(f"executed binding differs from formal selection: {target}/{name}")
            if not partial:
                if row.median_ms is None or not math.isfinite(row.median_ms) or row.median_ms <= 0 or not row.samples_ms:
                    raise ValueError(f"missing native measurement: {target}/{name}")
                require_jit_evidence(row, runtime, jit.get("disk_cache", {}))
                hot = jit.get("python_jit", {}).get("median_ms")
                if hot is None or not math.isfinite(hot) or hot <= 0:
                    raise ValueError(f"missing Python hot measurement: {target}/{name}")
            engines, families, resources = physical_evidence(metadata)
            record.update(
                cold_jit_ms=runtime.get("cold_compile_and_launch_ms", ""),
                compiler_api_ms=runtime.get("compiler_api_ms", ""),
                first_call_outside_compiler_ms=runtime.get("first_call_outside_compiler_ms", ""),
                cold_child_cpu_ms=runtime.get("cold_child_cpu_ms", ""),
                cache_hit_child_cpu_ms=runtime.get("cache_hit_child_cpu_ms", ""),
                first_cache_hit_ms=runtime.get("first_cache_hit_launch_ms", ""),
                disk_cache_restore_ms=jit.get("disk_cache", {}).get("cold_compile_and_launch_ms", ""),
                python_hot_ms=jit.get("python_jit", {}).get("median_ms", ""),
                python_hot_samples=len(runtime.get("samples_ms", [])),
                python_hot_protocol=runtime.get("python_measurement", {}).get("protocol", jit.get("measurement", {}).get("protocol", "")),
                disk_cache_hit=all(item["cache_hit"] for item in cache) if cache else "",
                same_compiled_kernel=runtime.get("same_compiled_kernel", ""),
                native_entry=row.details.get("entry", ""),
                used_extensions=json.dumps(metadata.get("kernel_abi", {}).get("used_extensions", [])),
                required_extensions=json.dumps(metadata.get("kernel_abi", {}).get("required_extensions", [])),
                instruction_path=row.details.get("runtime_instruction_path", "not_observed"),
                runtime_cpus=json.dumps(runtime.get("cpus", [])),
                tilelang_version=runtime.get("tilelang", ""),
                warmup=measure.get("warmup", ""), warmup_ms=measure.get("warmup_ms", 0.0),
                batch_ms=measure.get("batch_ms", ""), data_cache=measure.get("cache", ""),
                sample_order=measure.get("sample_order", "serial"), native_protocol=measure.get("protocol", ""),
                sample_count=len(timed_row.samples_ms),
                numerical_check_scope=row.correctness,
                selected_physical_engines=json.dumps(engines),
                instruction_families=json.dumps(families),
                resource_analysis=json.dumps(resources, sort_keys=True),
                target_arch=metadata.get("target", {}).get("arch", ""),
                target_vlen_bits=metadata.get("target", {}).get("vlen_bits", ""),
            )
            output.append(record)
    return output


def save(rows, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "operators.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for target in target_names():
        group = [row for row in rows if row["target"] == target]
        ratios = [row["reference_speedup"] for row in group if row["reference_speedup"] != ""]
        scopes = {}
        for scope in sorted({row["scope"] for row in group}):
            scoped = [row for row in group if row["scope"] == scope]
            matched = [row["reference_speedup"] for row in scoped if row["reference_speedup"] != ""]
            scopes[scope] = {
                "formal": len(scoped), "executed_and_correct": sum(row["status"] == "ok" for row in scoped),
                "matched_references": len(matched), "faster_than_reference": sum(value > 1 for value in matched),
                "median_speedup": statistics.median(matched) if matched else None,
            }
        summary[target] = {
            "formal": len(group), "statuses": dict(Counter(row["status"] for row in group)),
            "reference_matches": len(ratios), "reference_gaps": sum(row["reference_match"] == "catalog_gap" for row in group),
            "faster_than_reference": sum(value > 1 for value in ratios),
            "median_speedup": statistics.median(ratios) if ratios else None,
            "triton_paired": sum(row["triton_match"] == "paired" for row in group),
            "triton_within_3_percent": sum(row["parity_status"] == "passed" for row in group),
            "parity_statuses": dict(Counter(row["parity_status"] for row in group)),
            "slowest": sorted(({"case": row["case"], "speedup": row["reference_speedup"]} for row in group
                               if row["reference_speedup"] != ""), key=lambda item: item["speedup"])[:10],
            "by_scope": scopes,
        }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    complete = all(row["status"] == "ok" and row["parity_status"] == "passed"
                   and row["reference_match"] in ("matched", "catalog_gap") for row in rows)
    lines = ["# TileLang–Weft operator comparison / 算子对比", "",
             "Status / 状态: " + ("complete / 完成" if complete else "in progress / 进行中"), "",
             ("Speedup is matched native reference time divided by TileLang native time. "
              "倍率为匹配原生参考耗时除以 TileLang 原生耗时。"), "",
             "| Target / 平台 | Correct / 正确 | Matched references / 匹配参考 | Faster / 更快 | Median / 倍率中位数 |",
             "|---|---:|---:|---:|---:|"]
    for target, item in summary.items():
        median = f"{item['median_speedup']:.4g}×" if item["median_speedup"] is not None else "—"
        lines.append(f"| {target} | {item['statuses'].get('ok', 0)}/{item['formal']} | "
                     f"{item['reference_matches']} | {item['faster_than_reference']} | {median} |")
    lines += ["", "[Full operator CSV / 完整算子表](operators.csv)", "",
              "| Target / 平台 | Alternating Triton pairs / 交替配对 | Within 3% / 达标 |",
              "|---|---:|---:|"]
    for target, item in summary.items():
        lines.append(f"| {target} | {item['triton_paired']}/{item['formal']} | "
                     f"{item['triton_within_3_percent']}/{item['formal']} |")
    lines += ["",
              ("TileLang / Triton latency must be at most 1.03 for each case. "
               "每项 TileLang / Triton 耗时比须不超过 1.03。"), "",
              ("Native computation, cold JIT, cache restoration and Python hot calls have separate columns. "
               "原生计算、冷 JIT、缓存恢复和 Python 热调用分别记录。"), "",
              ("The CSV retains numerical-check scope, source parameters, hardware decisions and reference gaps. "
               "CSV 保留数值检查范围、源码参数、硬件决定和参考缺口。"), ""]
    for target, item in summary.items():
        lines += [f"## {target}", "",
                  "| Native workload boundary / 原生计算范围 | Correct / 正确 | Matched / 匹配 | Median / 中位数 |",
                  "|---|---:|---:|---:|"]
        for scope, scoped in item["by_scope"].items():
            median = f"{scoped['median_speedup']:.4g}×" if scoped["median_speedup"] is not None else "—"
            lines.append(f"| {scope} | {scoped['executed_and_correct']}/{scoped['formal']} | "
                         f"{scoped['matched_references']} | {median} |")
        lines += ["", "Slowest matched cases / 最慢的匹配用例：", ""]
        lines.extend(f"- `{row['case']}`: {row['speedup']:.4g}×" for row in item["slowest"])
        lines.append("")
    (destination / "summary.md").write_text("\n".join(lines))
    return summary


def public_observations(runs, references, triton, destination, paired):
    """Publish original samples without machine-private paths or replay inputs."""
    observations = []

    def retained(row):
        details = row.details
        execution = details.get("execution", {})
        environment = execution.get("env", {})
        jit = details.get("jit", {})
        runtime = jit.get("runtime", {})
        binding = None
        if jit.get("diagnostics"):
            path = Path(jit["diagnostics"]) / "jit-request.json"
            if path.is_file():
                request = json.loads(path.read_text())
                binding = {key: request[key] for key in (
                    "producer", "source_module", "symbol", "constexprs", "backend_options",
                    "grid", "num_cpu_threads",
                ) if key in request}
        return {
            **{key: getattr(row, key) for key in (
                "case", "implementation", "target", "shape", "scope", "status",
                "correctness", "median_ms", "samples_ms",
            )},
            "measurement": details.get("measurement", {}),
            "input_policy": details.get("input_policy"),
            "dtype": details.get("dtype"), "phase": details.get("phase"),
            "source_binding": binding,
            "producer": details.get("producer"),
            "producer_version": runtime.get("tilelang", runtime.get("triton")),
            "source_jit": {key: runtime[key] for key in (
                "status", "correctness", "cold_compile_and_launch_ms", "compiler_api_ms",
                "first_call_outside_compiler_ms", "cold_child_cpu_ms", "cache_hit_child_cpu_ms",
                "first_cache_hit_launch_ms", "same_compiled_kernel", "samples_ms",
                "python_measurement", "compilations",
            ) if key in runtime},
            "disk_cache": {key: jit.get("disk_cache", {})[key] for key in (
                "status", "correctness", "cold_compile_and_launch_ms", "compiler_api_ms",
                "cold_child_cpu_ms", "compilations",
            ) if key in jit.get("disk_cache", {})},
            "execution": {
                "cpus": execution.get("cpus", []), "threads": execution.get("threads"),
                "isa": execution.get("isa", {}),
                "memory_nodes": execution.get("numa", {}).get("memory_nodes", []),
                "environment": {key: environment[key] for key in (
                    "OMP_NUM_THREADS", "OMP_DYNAMIC", "ONEDNN_VERBOSE", "DNNL_VERBOSE",
                ) if key in environment},
            },
            "frequency_policies": [
                {"cpu": item["cpu"], "frequency_policy": item["frequency_policy"]}
                for item in details.get("observed_environment", {}).get("affinity", [])
                if item.get("frequency_policy")
            ],
        }

    for target, path in runs.items():
        catalog = target_catalog(target)["cases"]
        baseline = read(references[target]) if target in references else []
        comparison = read(triton[target]) if target in triton else []
        pairs = native_pairs(paired[target]) if target in paired else {}
        for name, row in observation_index(path).items():
            entry = {"tilelang": retained(row)}
            selected = catalog[name]
            if selected["reference"]:
                matches = [item for item in baseline if item.implementation == selected["reference"]
                           and item.status == "ok" and item.median_ms is not None and contracts_match(row, item)]
                if len(matches) != 1:
                    raise ValueError(f"public samples require one matched reference: {target}/{name}")
                entry["reference"] = retained(matches[0])
            matched_triton = [item for item in comparison if item.implementation == "weft"
                              and item.details.get("producer") == "triton"
                              and item.status == "ok" and item.median_ms is not None
                              and contracts_match(row, item)]
            if len(matched_triton) == 1:
                entry["triton_historical"] = retained(matched_triton[0])
            if name in pairs:
                candidate, control = checked_pair(row, pairs[name])
                entry["native_pair"] = {"tilelang": retained(candidate), "triton": retained(control)}
            observations.append(entry)
    (destination / "observations.json").write_text(json.dumps({
        "schema": 1,
        "scope": "original measured samples; private execution paths omitted; presentation export",
        "observations": observations,
    }, indent=2) + "\n")


def plot(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for target in target_names():
        values = sorted(row["reference_speedup"] for row in rows
                        if row["target"] == target and row["reference_speedup"] != "")
        if values:
            axes[0].step(values, [(i + 1) / len(values) for i in range(len(values))],
                         where="post", linewidth=1.7,
                         label=f"{target}: {len(values)} pairs, median {statistics.median(values):.3g}x")
    axes[0].axvline(1, color="#778899", linewidth=1, linestyle="--")
    axes[0].set(xscale="log", xlabel="Native reference / TileLang time", ylabel="Fraction of matched cases",
                title="Native performance distribution")
    axes[0].grid(axis="x", alpha=0.18)
    axes[0].legend(fontsize=8)
    labels = target_names()
    done = [sum(row["target"] == target and row["status"] == "ok" for row in rows) for target in labels]
    total = [sum(row["target"] == target for row in rows) for target in labels]
    axes[1].bar(labels, total, color="#dde4eb", label="Formal suite")
    axes[1].bar(labels, done, color="#3a7ba8", label="Executed and correct")
    axes[1].set(ylabel="Operator cases", title="Source JIT and numerical checks")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, max(total) * 1.14)
    for index, (count, formal) in enumerate(zip(done, total)):
        axes[1].text(index, formal + 2, f"{count}/{formal}", ha="center", fontsize=9)
    for suffix in ("svg", "png"):
        figure.savefig(destination / ("performance." + suffix), dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for target in labels:
        values = sorted(row["tilelang_over_triton"] for row in rows
                        if row["target"] == target and row["tilelang_over_triton"] != "")
        if values:
            axes[0].step(values, [(i + 1) / len(values) for i in range(len(values))],
                         where="post", linewidth=1.7, label=f"{target}: {len(values)} pairs")
    axes[0].axvline(1.03, color="#aa5b39", linewidth=1, linestyle="--", label="3% limit")
    axes[0].set(xscale="log", xlabel="TileLang native time / Triton native time",
                ylabel="Fraction of paired cases", title="Shared compiler performance inheritance")
    axes[0].grid(axis="x", alpha=0.18)
    axes[0].legend(fontsize=8)
    passed = [sum(row["target"] == target and row["parity_status"] == "passed" for row in rows) for target in labels]
    axes[1].bar(labels, total, color="#dde4eb", label="Formal suite")
    axes[1].bar(labels, passed, color="#397b64", label="Numerically correct and within 3%")
    axes[1].set(ylabel="Operator cases", title="Per-case Triton parity", ylim=(0, max(total) * 1.14))
    axes[1].legend(fontsize=8)
    for index, (count, formal) in enumerate(zip(passed, total)):
        axes[1].text(index, formal + 2, f"{count}/{formal}", ha="center", fontsize=9)
    for suffix in ("svg", "png"):
        figure.savefig(destination / ("parity." + suffix), dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "reference", "triton", "paired"):
        parser.add_argument("--" + name, action="append", default=[], metavar="TARGET=RUN")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="retain missing/failed rows as explicit incomplete states")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    output = OutputRun(args.output)
    try:
        runs, references = mapping(args.run), mapping(args.reference)
        triton = mapping(args.triton)
        paired = mapping(args.paired)
        rows = records(runs, references, triton, partial=args.partial, paired=paired)
        summary = save(rows, output.stage)
        if not args.partial:
            public_observations(runs, references, triton, output.stage, paired)
        if args.plot:
            plot(rows, output.stage)
    except Exception as error:
        output.finish(error=error)
        raise
    output.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
