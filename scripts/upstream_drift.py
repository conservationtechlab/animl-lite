"""
Report functions that changed in animl-py since animl-lite last synced with them

animl-lite is an onnx-only fork of animl-py, so many functions differ on purpose.
upstream_sync.json records each animl-py function as it was at the last review.
A function drifts when animl-py has changed it since then and it doesn't match
animl-lite. Functions and modules that are new in animl-py are reported too.

Usage:
    python scripts/upstream_drift.py --upstream ../animl-py           # report, exit 1 on drift
    python scripts/upstream_drift.py --upstream ../animl-py --update  # record the reviewed state
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "upstream_sync.json"


def functions(path: Path) -> dict:
    """
    Normalized source of each function and method in a file, ignoring formatting and docstrings.

    Args:
        path (Path): python file

    Returns:
        dict of 'function' or 'Class.method' to a hash of its normalized source
    """
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        defs = [(node.name, node)] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
        if isinstance(node, ast.ClassDef):
            defs = [(f"{node.name}.{n.name}", n) for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for name, fn in defs:
            body = fn.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body = body[1:]  # drop the docstring
            fn = ast.FunctionDef(fn.name, fn.args, body, fn.decorator_list, fn.returns, None, [])
            out[name] = hashlib.sha256(ast.dump(fn).encode()).hexdigest()[:16]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", required=True, help="path to an animl-py checkout")
    parser.add_argument("--update", action="store_true", help="record the current animl-py state as reviewed")
    args = parser.parse_args()

    lite_src = ROOT / "src" / "animl"
    py_src = Path(args.upstream) / "src" / "animl"
    lite_modules = {str(p.relative_to(lite_src)) for p in lite_src.rglob("*.py")}
    py_modules = {str(p.relative_to(py_src)) for p in py_src.rglob("*.py")}

    current = {}  # 'module:function' -> animl-py hash, or 'module' -> 'upstream-only module'
    report = {"drift": [], "new": [], "same": 0, "accepted": 0}
    baseline = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {}

    for module in sorted(py_modules):
        if module not in lite_modules:
            current[module] = "upstream-only module"
            if module not in baseline:
                report["new"].append(module)
            continue
        py_fns = functions(py_src / module)
        lite_fns = functions(lite_src / module)
        for name, py_hash in py_fns.items():
            key = f"{module}:{name}"
            current[key] = py_hash
            if lite_fns.get(name) == py_hash:
                report["same"] += 1
            elif key not in baseline:
                report["new"].append(key)
            elif baseline[key] != py_hash:
                report["drift"].append(key)
            else:
                report["accepted"] += 1

    if args.update:
        BASELINE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"Recorded {len(current)} animl-py functions and modules in {BASELINE.name}")
        return 0

    print(f"{report['same']} functions match animl-py, {report['accepted']} differ as reviewed")
    for label, items in [("Changed in animl-py since the last review", report["drift"]),
                         ("New in animl-py, not reviewed yet", report["new"])]:
        if items:
            print(f"\n{label}:")
            for item in items:
                print(f"  {item}")
    if report["drift"] or report["new"]:
        print("\nPort or skip these, then run with --update to record the review.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
