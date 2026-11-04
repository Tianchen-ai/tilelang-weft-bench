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
    "python_hot_ms", "disk_cache_hit", "same_compiled_kernel", "native_entry",
    "source", "symbol", "source_parameters", "backend_options", "used_extensions",
    "required_extensions", "selected_physical_engines", "target_arch", "target_vlen_bits",
    "instruction_path", "runtime_cpus", "tilelang_version",
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
            if not partial:
                if row.median_ms is None or not math.isfinite(row.median_ms) or row.median_ms <= 0 or not row.samples_ms:
                    raise ValueError(f"missing native measurement: {target}/{name}")
                if runtime.get("same_compiled_kernel") is not True or not cache or not all(item["cache_hit"] for item in cache):
                    raise ValueError(f"missing source JIT/cache evidence: {target}/{name}")
                hot = jit.get("python_jit", {}).get("median_ms")
                if hot is None or not math.isfinite(hot) or hot <= 0:
                    raise ValueError(f"missing Python hot measurement: {target}/{name}")
            record.update(
                cold_jit_ms=runtime.get("cold_compile_and_launch_ms", ""),
                first_cache_hit_ms=runtime.get("first_cache_hit_launch_ms", ""),
                python_hot_ms=jit.get("python_jit", {}).get("median_ms", ""),
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
                selected_physical_engines=json.dumps(sorted({
                    operation["engine"]
                    for operation in row.details.get("physical_program", {}).get("physical_operations", {}).get("operations", [])
                    if operation.get("engine")
                })),
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
