"""TileLang bindings for the fixed Weft operator workload contracts."""

import json
from importlib.util import find_spec
from pathlib import Path

from weft_bench.catalogs import references as references
from weft_bench.catalogs import target_catalog as reference_catalog
from weft_bench.catalogs import target_names as target_names

CANDIDATE = "tilelang"
FACTORY = "weft_bench.adapters.triton_jit:TileLangJitAdapter"
ROOT = Path(__file__).parent


def target_catalog(name):
    selected = reference_catalog(name)
    bindings = json.loads((ROOT / "targets" / (name + ".json")).read_text())
    protocols = json.loads((ROOT / "measurement.json").read_text())
    overrides = protocols["targets"][name]
    if set(bindings) != set(selected["cases"]):
        raise ValueError("TileLang bindings differ from the fixed workload catalog")
    if set(overrides) - set(selected["cases"]):
        raise ValueError("timing policy names are outside the formal workload catalog")
    for case, record in selected["cases"].items():
        record["candidate"] = bindings[case]
        record["measurement"] = {**protocols["defaults"], **overrides.get(case, {})}
        record["artifact_evidence"] = {"used_extensions": [], "matrix_in_artifact": False,
                                       "runtime_instruction_path": "not_observed"}
    selected.pop("implementations", None)
    selected["suites"]["full"] = list(selected["cases"])
    return selected


def resolve_source(source):
    if not source.startswith("module:"):
        raise ValueError("TileLang suite source must identify an installed product module")
    module = source.removeprefix("module:")
    if not module.startswith("tilelang.weft.kernels."):
        raise ValueError("suite source must belong to tilelang.weft.kernels")
    spec = find_spec(module)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(module)
    return str(Path(spec.origin).resolve(strict=True))


def models():
    return {}
