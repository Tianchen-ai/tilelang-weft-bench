"""Present saved engine observations; no compilation, execution or timing."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from weft_bench.results import apply_baselines, contracts_match, read

from .catalogs import target_catalog, target_names

FIELDS = (
    "target", "case", "shape", "scope", "status", "correctness", "native_us",
    "reference", "reference_us", "reference_speedup", "reference_match",
    "triton_native_us", "triton_match", "cold_jit_ms", "first_cache_hit_ms",
    "python_hot_ms", "python_hot_samples", "python_hot_protocol", "disk_cache_hit", "same_compiled_kernel", "native_entry",
    "source", "symbol", "source_parameters", "backend_options", "binding_match", "used_extensions",
    "required_extensions", "selected_physical_engines", "target_arch", "target_vlen_bits",
    "instruction_path", "instruction_families", "resource_analysis", "runtime_cpus", "tilelang_version",
    "warmup", "batch_ms", "sample_count", "numerical_check_scope",
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


def records(runs, references, triton, *, partial):
    output = []
    for target in target_names():
        catalog = target_catalog(target)
        current = observation_index(runs[target]) if target in runs else {}
        baseline = read(references[target]) if target in references else []
        old = read(triton[target]) if target in triton else []
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
                          native_us=row.median_ms * 1000 if row.median_ms is not None else "")
            if row.status != "ok" and not partial:
                raise ValueError(f"unresolved execution: {target}/{name}: {row.status}")
            if selected["reference"]:
                apply_baselines([row], baseline, selected["reference"])
                record["reference_match"] = "matched" if row.speedup is not None else "unmatched"
                if row.speedup is not None:
                    record.update(reference_us=row.details["baseline_median_ms"] * 1000,
                                  reference_speedup=row.speedup)
                elif not partial:
                    raise ValueError(f"unmatched required reference: {target}/{name}: {row.details.get('comparison_note')}")
            else:
                record["reference_match"] = "catalog_gap"
            matches = [item for item in old if item.implementation == "weft" and item.status == "ok"
                       and item.median_ms is not None and contracts_match(row, item)]
            record["triton_match"] = "matched" if len(matches) == 1 else "unavailable" if not matches else "ambiguous"
            if len(matches) == 1:
                record["triton_native_us"] = matches[0].median_ms * 1000
            jit = row.details.get("jit", {})
            runtime = jit.get("runtime", {})
            cache = jit.get("disk_cache", {}).get("compilations", [])
            measure = row.details.get("measurement", {})
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
                if runtime.get("same_compiled_kernel") is not True or not cache or not all(item["cache_hit"] for item in cache):
                    raise ValueError(f"missing source JIT/cache evidence: {target}/{name}")
                hot = jit.get("python_jit", {}).get("median_ms")
                if hot is None or not math.isfinite(hot) or hot <= 0:
                    raise ValueError(f"missing Python hot measurement: {target}/{name}")
            engines, families, resources = physical_evidence(metadata)
            record.update(
                cold_jit_ms=runtime.get("cold_compile_and_launch_ms", ""),
                first_cache_hit_ms=runtime.get("first_cache_hit_launch_ms", ""),
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
                warmup=measure.get("warmup", ""), batch_ms=measure.get("batch_ms", ""),
                sample_count=len(row.samples_ms),
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
        summary[target] = {
            "formal": len(group), "statuses": dict(Counter(row["status"] for row in group)),
            "reference_matches": len(ratios), "reference_gaps": sum(row["reference_match"] == "catalog_gap" for row in group),
            "faster_than_reference": sum(value > 1 for value in ratios),
            "median_speedup": statistics.median(ratios) if ratios else None,
            "slowest": sorted(({"case": row["case"], "speedup": row["reference_speedup"]} for row in group
                               if row["reference_speedup"] != ""), key=lambda item: item["speedup"])[:10],
        }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    complete = all(row["status"] == "ok" and row["reference_match"] in ("matched", "catalog_gap") for row in rows)
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
              ("Native computation, cold JIT, cache restoration and Python hot calls have separate columns. "
               "原生计算、冷 JIT、缓存恢复和 Python 热调用分别记录。"), "",
              ("The CSV retains numerical-check scope, source parameters, hardware decisions and reference gaps. "
               "CSV 保留数值检查范围、源码参数、硬件决定和参考缺口。"), ""]
    (destination / "summary.md").write_text("\n".join(lines))
    return summary


def plot(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for target in target_names():
        values = sorted(row["reference_speedup"] for row in rows
                        if row["target"] == target and row["reference_speedup"] != "")
        if values:
            axes[0].step(values, [(i + 1) / len(values) for i in range(len(values))], label=f"{target} (n={len(values)})")
    axes[0].axvline(1, color="#778899", linewidth=1, linestyle="--")
    axes[0].set(xscale="log", xlabel="Native reference / TileLang time", ylabel="Fraction of matched cases")
    axes[0].legend(fontsize=8)
    labels = target_names()
    done = [sum(row["target"] == target and row["status"] == "ok" for row in rows) for target in labels]
    total = [sum(row["target"] == target for row in rows) for target in labels]
    axes[1].bar(labels, total, color="#dde4eb", label="Formal suite")
    axes[1].bar(labels, done, color="#3a7ba8", label="Executed and correct")
    axes[1].set(ylabel="Operator cases")
    axes[1].legend(fontsize=8)
    for suffix in ("svg", "png"):
        figure.savefig(destination / ("performance." + suffix), dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "reference", "triton"):
        parser.add_argument("--" + name, action="append", default=[], metavar="TARGET=RUN")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="retain missing/failed rows as explicit incomplete states")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    rows = records(mapping(args.run), mapping(args.reference), mapping(args.triton), partial=args.partial)
    summary = save(rows, args.output)
    if args.plot:
        plot(rows, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
