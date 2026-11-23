"""
Compare animl-lite's modules with animl-py's, side by side in VS Code

animl-py's version is exported (read-only) to .upstream/, and the diff opens with it
on the left and animl-lite's own file on the right, so you can align animl-lite as
you read. --aligned instead lines both files up function by function, as functions
are often in a different order; that view is for reading only.

Usage:
    python scripts/upstream_diff.py                       # list modules and how much they differ
    python scripts/upstream_diff.py classification        # open one module (partial names work)
    python scripts/upstream_diff.py detection reid        # open several
    python scripts/upstream_diff.py --all                 # open every module that differs
    python scripts/upstream_diff.py reid --aligned        # function-by-function view
    python scripts/upstream_diff.py --ref upstream/main   # compare with another animl-py ref
"""
import argparse
import ast
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LITE = ROOT / "src" / "animl"
OUT = ROOT / ".upstream"


def export(ref: str, fetch: bool) -> Path:
    """
    Export animl-py's src/animl at ref to .upstream/, read-only.

    Args:
        ref (str): git ref of animl-py, e.g. upstream/dev
        fetch (bool): fetch the remote first

    Returns:
        path to the exported animl package
    """
    if fetch and "/" in ref:
        subprocess.run(["git", "fetch", "-q", ref.split("/")[0]], cwd=ROOT, check=True)
    dest = OUT / ref.replace("/", "_")
    if dest.exists():
        for p in dest.rglob("*"):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", ref, "src/animl"], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    for p in dest.rglob("*.py"):
        p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return dest / "src" / "animl"


def changed_lines(a: Path, b: Path) -> int:
    """
    Number of changed lines between two files.

    Args:
        a (Path): first file
        b (Path): second file

    Returns:
        added plus removed lines
    """
    out = subprocess.run(["git", "diff", "--no-index", "--numstat", str(a), str(b)], capture_output=True, text=True)
    return sum(int(n) for line in out.stdout.splitlines() for n in line.split()[:2] if n.isdigit())


def aligned(py_file: Path, lite_file: Path, module: str) -> tuple:
    """
    Write both files with their top-level functions and classes in the same order.

    Definitions follow animl-py's order, then animl-lite-only ones; a definition missing
    on one side gets a placeholder comment there, so the diff lines up by function.

    Args:
        py_file (Path): animl-py module
        lite_file (Path): animl-lite module
        module (str): module path, e.g. utils/general.py, written under .upstream/aligned/

    Returns:
        tuple of the aligned animl-py and animl-lite files
    """
    def blocks(path):
        text = path.read_text().splitlines(keepends=True)
        tree = ast.parse("".join(text))
        defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        header_end = (min(d.decorator_list[0].lineno if d.decorator_list else d.lineno for d in defs) - 1) if defs else len(text)
        out = {}
        for d in defs:
            start = (d.decorator_list[0].lineno if d.decorator_list else d.lineno) - 1
            out[d.name] = "".join(text[start:d.end_lineno]).rstrip() + "\n"
        return "".join(text[:header_end]).rstrip() + "\n", out

    py_header, py_defs = blocks(py_file)
    lite_header, lite_defs = blocks(lite_file)
    names = list(py_defs) + [n for n in lite_defs if n not in py_defs]
    files = []
    for label, header, defs, other in [("animl-py", py_header, py_defs, "animl-py"),
                                       ("animl-lite", lite_header, lite_defs, "animl-lite")]:
        parts = [header] + [defs.get(n, f"# {n}: not in {other}\n") for n in names]
        path = OUT / "aligned" / label / module
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        path.write_text("\n\n".join(parts))
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        files.append(path)
    return tuple(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modules", nargs="*", help="modules to open, e.g. classification or utils/general")
    parser.add_argument("--all", action="store_true", help="open every module that differs")
    parser.add_argument("--aligned", action="store_true", help="line functions up by name (read-only view)")
    parser.add_argument("--ref", default="upstream/dev", help="animl-py ref to compare with (default upstream/dev)")
    parser.add_argument("--no-fetch", action="store_true", help="don't fetch the animl-py remote first")
    args = parser.parse_args()

    py = export(args.ref, fetch=not args.no_fetch)
    lite_modules = {str(p.relative_to(LITE)) for p in LITE.rglob("*.py")}
    py_modules = {str(p.relative_to(py)) for p in py.rglob("*.py")}
    shared = sorted(lite_modules & py_modules)
    counts = {m: changed_lines(py / m, LITE / m) for m in shared}

    if not args.modules and not args.all:
        print(f"animl-lite vs animl-py {args.ref} (changed lines)\n")
        for m in sorted(shared, key=lambda m: -counts[m]):
            print(f"  {counts[m]:5d}  {m}" if counts[m] else f"   same  {m}")
        print("\nOnly in animl-lite: " + (", ".join(sorted(lite_modules - py_modules)) or "-"))
        print("Only in animl-py:   " + (", ".join(sorted(py_modules - lite_modules)) or "-"))
        print("\nOpen one with: python scripts/upstream_diff.py <module>")
        return 0

    if args.all:
        selected = [m for m in shared if counts[m]]
    else:
        selected = []
        for name in args.modules:
            name = name.split("src/animl/")[-1]  # a path to the file works too
            matches = [m for m in shared if name in (m, m[:-3])] or \
                      [m for m in shared if Path(m).stem == name] or \
                      [m for m in shared if name in m]
            if not matches:
                print(f"No module in both packages matches '{name}'")
                return 1
            selected += matches

    for m in dict.fromkeys(selected):
        left, right = py / m, LITE / m
        if args.aligned:
            left, right = aligned(left, right, m)
        subprocess.run(["code", "--diff", str(left), str(right)], check=True)
        print(f"Opened {m} ({counts[m]} changed lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
