"""Prepare a local OpenBLAS dependency, then explicitly run the native operator checks."""
import argparse
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if (ROOT / "cli/crit").is_dir():
    sys.path[:0] = [str(ROOT / "cli"), str(ROOT / "kernels")]
import crit_kernels as kernels
from crit.process.ownership import RuntimeOwnership
from crit.process.tree import guarded_tree
from crit.run.records import file_sha, inventory, read_json, safe_path, text, write_json


def invocation(value):
    kernels.header(value, "openblas_invocation", {"library", "integer_bits", "symbols"})
    if type(value["integer_bits"]) is not int or value["integer_bits"] not in {32, 64}:
        raise ValueError("OpenBLAS integer_bits must be 32 or 64")
    kernels.fields(value["symbols"], {"sgemm", "config", "threads"})
    for name in value["symbols"].values():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name):
            raise ValueError("OpenBLAS symbols require bounded ASCII identifiers")
    return value


def host():
    machine = platform.machine().lower()
    return {"os": platform.system().lower(), "arch": {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine), "backend": "cpu"}


def prepare(args):
    work = args.work.absolute()
    if work.exists() or work.resolve() != work:
        raise ValueError("Choose a new work directory without symlink/reparse traversal")
    spec = invocation({"schema_version": 1, "record_kind": "openblas_invocation",
        "library": "lib/" + args.library.name, "integer_bits": args.integer_bits,
        "symbols": {"sgemm": args.sgemm_symbol, "config": args.config_symbol, "threads": args.threads_symbol}})
    sources = [(args.library, spec["library"], "binary")]
    sources += [(p, "lib/" + p.name, "runtime") for p in args.runtime_file]
    sources += [(p, f"notices/{i}-{p.name}", "notice") for i, p in enumerate(args.notice)]
    sources += [(p, f"metadata/{i}-{p.name}", "metadata") for i, p in enumerate(args.metadata)]
    artifacts = []
    for source, name, role in sources:
        source = source.absolute()
        checked = safe_path(source.parent, source.name)
        if not checked.is_file():
            raise ValueError("Dependency input must be a regular file: " + str(source))
        artifacts.append({"path": name, "role": role, "bytes": source.stat().st_size, "sha256": file_sha(source)})
    if sum(item["bytes"] for item in artifacts) > 8 * 1024 ** 3:
        raise ValueError("Example dependency exceeds its 8 GiB byte limit")
    abi = "cblas-ilp64" if args.integer_bits == 64 else "cblas-lp64"
    req = {"schema_version": 1, "record_kind": "kernels_request", "operation": {"name": "gemm",
        "dtype": "float32", "layout": "row-major", "m": 7, "n": 11, "k": 16, "transpose_a": False,
        "transpose_b": False, "aliasing": "disjoint", "alpha": 1.25, "beta": -0.5}, "target": host() | {"abi": abi}}
    requirements = {space + "." + key: {"one_of": [value]} for space in ("operation", "target") for key, value in req[space].items()}
    for name in ("m", "n", "k"):
        requirements["operation." + name] = {"minimum": 1, "maximum": 2 ** (args.integer_bits - 1) - 1}
    for name in ("transpose_a", "transpose_b"):
        requirements["operation." + name] = {"one_of": [False, True]}
    for name in ("alpha", "beta"):
        requirements["operation." + name] = {"minimum": -2 ** 63, "maximum": 2 ** 63}
    entry = {"schema_version": 1, "record_kind": "kernels_implementation", "id": "openblas.sgemm",
        "summary": "Local OpenBLAS SGEMM: explicit native ABI, row-major leading dimensions and optional transposition",
        "provenance": {"repository": args.repository, "revision": args.revision or "sha256:" + artifacts[0]["sha256"], "variant": args.variant},
        "license": args.license, "requirements": requirements, "artifacts": artifacts,
        "contract": {"semantics": "C = alpha*op(A)*op(B) + beta*C, float32; positive dimensions, finite alpha/beta, disjoint output, valid row leading dimensions.",
            "invocation": "C ABI; symbols and integer width in openblas.json. CBLAS enums are 32-bit. Reject mismatched OpenBLAS USE64BITINT before matrix calls.",
            "workspace": "OpenBLAS owns its internal CPU workspace/thread pool. The caller allocates all input/output matrices and budgets provider workspace separately.",
            "completion": "Synchronous CBLAS return settles reads/writes for this call. The example selects one provider thread.",
            "ownership": "Caller retains the library, inputs and output through return; closes only after all calls settle. Thread settings affect other users of the same loaded library."},
        "runtime_dependencies": args.runtime_requirement, "unverified": []}
    kernels.implementation(entry)  # Validate names/notices before copying inputs.
    work.mkdir(parents=True)
    source_root = work / "provider"
    for (source, name, _), artifact in zip(sources, artifacts):
        destination = safe_path(source_root, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_sha(destination) != artifact["sha256"]:
            raise ValueError("Input changed during preparation: " + str(source))
    write_json(source_root / "openblas.json", spec, immutable=True)
    entry["artifacts"].append({"path": "openblas.json", "role": "metadata",
        "bytes": (source_root / "openblas.json").stat().st_size, "sha256": file_sha(source_root / "openblas.json")})
    write_json(work / "catalog/openblas.json", entry, immutable=True)
    write_json(work / "request.json", req, immutable=True)
    kernels.retain(req, {entry["id"]: entry}, entry["id"], source_root, work / "dependency",
                  "Selected the explicitly supplied OpenBLAS binary and ABI for native operator validation")
    return kernels.verify(work / "dependency")


def check(dependency, executable, output, timeout=30, *, profile="gemm"):
    if profile not in {"gemm", "attention"}:
        raise ValueError("Choose the gemm or attention check profile")
    if type(timeout) not in {int, float} or not 0 < timeout <= 300 or not math.isfinite(timeout):
        raise ValueError("Native timeout must be finite and between 0 and 300 seconds")
    dependency, executable, output = (Path(p).absolute() for p in (dependency, executable, output))
    before = kernels.verify(dependency)
    spec = invocation(read_json(safe_path(dependency, "artifacts/openblas.json")))
    library = safe_path(dependency, "artifacts/" + spec["library"])
    req = read_json(safe_path(dependency, "request.json"))
    if any(req["target"].get(k) != value for k, value in host().items()):
        raise ValueError("Retained dependency targets another host OS, architecture or backend")
    expected_abi = "cblas-ilp64" if spec["integer_bits"] == 64 else "cblas-lp64"
    if req["target"].get("abi") != expected_abi or any(req["operation"].get(k) != v for k, v in
            {"name": "gemm", "dtype": "float32", "layout": "row-major"}.items()):
        raise ValueError("Invocation differs from retained operation/ABI")
    entries = read_json(dependency / "catalog.json")
    selected = next(entry for entry in entries if entry["id"] == before["selected"])
    if not any(item["path"] == spec["library"] and item["role"] == "binary" for item in selected["artifacts"]):
        raise ValueError("Invocation library must be an inventoried selected binary")
    safe_path(executable.parent, executable.name)
    executable_sha = file_sha(executable)
    if output.exists() or output.resolve() != output or output.is_relative_to(dependency):
        raise ValueError("Choose a new result directory outside the dependency without reparse traversal")
    command = [str(executable), *(["openblas"] if profile == "attention" else []), str(library),
        str(spec["integer_bits"]), *[spec["symbols"][k] for k in ("sgemm", "config", "threads")]]
    owner = RuntimeOwnership()
    owner.acquire()
    tree, error, native, created = None, None, None, False
    settled = True
    try:
        output.mkdir(parents=True)
        created = True
        with (output / "native.log").open("xb") as log:
            tree = guarded_tree(command, output, log, owner, output)
            deadline = time.monotonic() + timeout
            while tree.process.poll() is None:
                if time.monotonic() >= deadline:
                    raise ValueError("Native operator check timed out")
                if (output / "native.log").stat().st_size > 1_048_576:
                    raise ValueError("Native operator check exceeded its diagnostic limit")
                time.sleep(.01)
            if tree.process.returncode:
                raise ValueError("Native operator check failed; see native.log")
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
    finally:
        try:
            if tree:
                tree.close()
            if owner.failed_tree:
                owner.failed_tree.close()
            owner.release(settled=True)
        except BaseException as exc:
            settled = False
            owner.block(str(exc))
            error = "Native process cleanup is unresolved: " + str(exc)
    if error is None:
        try:
            if (output / "native.log").stat().st_size > 1_048_576:
                raise ValueError("Native diagnostic limit exceeded")
            native = read_json(output / "native.log")
            extra = {"integration_cases", "max_output_error", "max_logsumexp_error"} if profile == "attention" else set()
            kernels.header(native, "attention_check" if profile == "attention" else "openblas_check",
                          {"cases", "integer_bits", "configuration"} | extra)
            text(native["configuration"], "Native configuration", 4096)
            cases = 2304 if profile == "attention" else 648
            if type(native["cases"]) is not int or native["cases"] != cases or type(native["integer_bits"]) is not int or native["integer_bits"] != spec["integer_bits"]:
                raise ValueError("Native check coverage/ABI differs from the required matrix")
            if profile == "attention":
                if type(native["integration_cases"]) is not int or native["integration_cases"] != 8:
                    raise ValueError("Native attention integration coverage differs")
                for name in ("max_output_error", "max_logsumexp_error"):
                    if type(native[name]) not in {int, float} or not math.isfinite(native[name]) or not 0 <= native[name] <= .001:
                        raise ValueError("Native attention error exceeds its bounded fixture limit")
            if kernels.verify(dependency) != before or file_sha(executable) != executable_sha:
                raise ValueError("Checker or retained dependency changed during execution")
        except (ValueError, OSError, TypeError, KeyError) as exc:
            error, native = str(exc), None
    scope = ("2304 bounded paged attention cases and 8 RoPE/prefill/decode/COW integration comparisons; independent double reference, masks, GQA/MQA and input immutability."
             if profile == "attention" else "648 bounded SGEMM cases; double reference, input immutability and output padding/guards.")
    result = {"schema_version": 1, "record_kind": "openblas_attention_validation" if profile == "attention" else "openblas_validation", "status": "passed" if error is None else "failed",
        "dependency_sha256": before["dependency_sha256"], "executable_sha256": executable_sha,
        "command": command, "native": native, "error": error, "cleanup_settled": settled,
        "scope": scope + " No model, speed or full-engine qualification."}
    if created:
        result["evidence"] = inventory(output)
        write_json(output / "result.json", result, immutable=True)
    if error:
        raise ValueError(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subs.add_parser("prepare", help="Copy explicitly supplied files; never load/download the provider")
    prepare_parser.add_argument("--library", type=Path, required=True)
    prepare_parser.add_argument("--notice", type=Path, action="append", required=True)
    prepare_parser.add_argument("--runtime-file", type=Path, action="append", default=[])
    prepare_parser.add_argument("--metadata", type=Path, action="append", default=[])
    prepare_parser.add_argument("--runtime-requirement", action="append", default=[])
    prepare_parser.add_argument("--work", type=Path, required=True)
    prepare_parser.add_argument("--integer-bits", type=int, choices=[32, 64], required=True)
    for name in ("sgemm", "config", "threads"):
        prepare_parser.add_argument("--" + name + "-symbol", required=True)
    prepare_parser.add_argument("--repository", default="https://github.com/OpenMathLib/OpenBLAS")
    prepare_parser.add_argument("--revision", help="Exact source revision; default pins the supplied binary's SHA-256")
    prepare_parser.add_argument("--variant", required=True)
    prepare_parser.add_argument("--license", required=True)
    check_parser = subs.add_parser("check", help="Explicitly execute the compiled check against verified local dependency bytes")
    check_parser.add_argument("--profile", choices=("gemm", "attention"), default="gemm")
    for name in ("dependency", "executable", "output"):
        check_parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        value = prepare(args) if args.command == "prepare" else check(args.dependency, args.executable, args.output, profile=args.profile)
        print(json.dumps(value, indent=2, ensure_ascii=True))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
