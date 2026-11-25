"""Grade the puzzles under syncsan.

    python grade.py                  grade every puzzle in puzzles/
    python grade.py 3 7              grade puzzles 3 and 7
    python grade.py --dir solutions  grade the reference solutions
    python grade.py --validate       check that every puzzle fails as given and every solution passes

A puzzle passes when four things hold: its output is correct when run natively, syncsan reports no racing pairs,
its output is correct in adversarial mode (each racing load returns the value from before the write it raced with), and
its output is correct in the control (the same loads redirected, with current values). The cost is the number of
synchronization operations you wrote inside the edit markers; par is the reference solution's.

syncsan is found through $SYNCSAN_SO, or `syncsan where` (pip install syncsan, then `syncsan build --arch 90`).
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "puzzles" / "manifest.json"
BEGIN, END = "// >>> edit", "// <<< edit"
RESULT = re.compile(r"^(\w+: (correct|WRONG)|needs sm_|cuda\w*\(|.*: (an illegal|unspecified launch|.*error))")

# What counts toward cost: barriers, fences, and memory-order qualifiers. Plain atomics are free.
COST = re.compile(r"\b(__syncthreads\w*|__syncwarp|__threadfence\w*|fence_\w+|bar_sync|bar_arrive|st_release|ld_acquire"
                  r"|atom_\w+_(?:release|acquire|acq_rel)|memory_order_(?:acquire|release|acq_rel|seq_cst)"
                  r"|atomic_thread_fence)\b"
                  r"|\b(?:fence|bar|barrier|mbarrier)\.\w+|\b(?:ld|st|atom|red)\.(?:release|acquire|acq_rel)\b")


def regions(text):
    """(skeleton, [region text]): the file with each edit region emptied, and what each region holds."""
    out, regs, cur = [], [], None
    for line in text.splitlines():
        s = line.strip()
        if s == BEGIN:
            if cur is not None:
                raise SystemExit("an edit region opens inside another")
            cur = []
            out.append(s)
        elif s == END:
            if cur is None:
                raise SystemExit("an edit region closes that never opened")
            regs.append("\n".join(cur))
            cur = None
            out.append(s)
        elif cur is not None:
            cur.append(line)
        else:
            out.append(line.rstrip())
    if cur is not None:
        raise SystemExit("an edit region never closes")
    return "\n".join(out), regs


def strip_comments(code):
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    return re.sub(r"//[^\n]*", " ", code)


def cost(regs):
    return sum(len(COST.findall(strip_comments(r))) for r in regs)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def puzzle_files(d):
    return sorted(p for p in (ROOT / d).glob("[0-9][0-9]_*.cu"))


def make_manifest():
    """Record each puzzle's skeleton, par (the solution's cost) and minimum architecture."""
    m = {}
    for p in puzzle_files("puzzles"):
        skel, _ = regions(p.read_text())
        sol = ROOT / "solutions" / p.name
        sskel, sregs = regions(sol.read_text())
        if sskel != skel:
            raise SystemExit(f"{sol.name}: the solution changes code outside the edit markers")
        m[p.name] = {"skeleton": digest(skel), "par": cost(sregs), "min_arch": 90 if "sm_90 and newer" in skel else 80}
    MANIFEST.write_text(json.dumps(m, indent=1) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(m)} puzzles)")


def gpu_arch():
    if os.environ.get("ARCH"):
        return int(os.environ["ARCH"])
    try:
        cc = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                            text=True, check=True).stdout.split()[0]
        return int(cc.replace(".", ""))
    except Exception:
        raise SystemExit("no NVIDIA GPU found (nvidia-smi failed); set ARCH=90 to compile without one")


def syncsan_so():
    if os.environ.get("SYNCSAN_SO"):
        return os.environ["SYNCSAN_SO"]
    exe = shutil.which("syncsan")
    if exe:
        p = subprocess.run([exe, "where"], capture_output=True, text=True).stdout.strip()
        if p.endswith(".so") and os.path.exists(p):
            return p
    raise SystemExit("syncsan not found: pip install git+https://github.com/ShreshthRajan/syncsan, then "
                     "`syncsan build --arch <your GPU's, e.g. 90>`, or set SYNCSAN_SO")


def run(cmd, env=None, timeout=600):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return None, out


def verdict_line(code, out):
    """The program's own result line, or why there is none."""
    if code is None:
        return False, "timed out (a barrier or a spin that never completes?)"
    lines = [l.strip() for l in out.splitlines() if RESULT.match(l.strip()) and not l.startswith("syncsan")]
    msg = "; ".join(lines) or f"exit status {code}" + "".join(f": {l.strip()}" for l in out.splitlines()[-1:])
    return code == 0, msg


def race_reports(out):
    """Each RACE report as its first three lines (the two accesses), in order."""
    lines, reps = out.splitlines(), []
    for i, l in enumerate(lines):
        if l.startswith("syncsan: RACE"):
            reps.append([l] + [x for x in lines[i + 1:i + 3]])
    return reps


def grade(path, arch, so, verbose):
    name = path.name
    man = json.loads(MANIFEST.read_text())[name]
    skel, regs = regions(path.read_text())
    title = path.read_text().splitlines()[0].lstrip("/ ").strip()
    print(f"\n{title}  ({path.parent.name}/{name})")
    if digest(skel) != man["skeleton"]:
        print("  code outside the edit markers changed: put it back (git checkout) and edit only between them")
        return "FAIL"
    if arch < man["min_arch"]:
        print(f"  skipped: needs sm_{man['min_arch']} or newer, this GPU is sm_{arch}")
        return "SKIP"
    build = ROOT / "build" / path.parent.name
    build.mkdir(parents=True, exist_ok=True)
    exe = build / path.stem
    code, out = run(["nvcc", "-O2", "-lineinfo", f"-arch=sm_{arch}", "-I", str(ROOT / "puzzles"), "-o", str(exe),
                     str(path)], timeout=300)
    if code != 0:
        print("  does not compile:\n" + "\n".join("    " + l for l in out.strip().splitlines()[:20]))
        return "FAIL"
    ok = True

    code, out = run([str(exe)], timeout=60)
    good, msg = verdict_line(code, out)
    print(f"  native       {msg}")
    ok &= good
    if code is None:
        return "FAIL"

    env = dict(os.environ, LD_PRELOAD=so)
    env.pop("SYNCSAN_ADVERSARIAL", None)
    code, out = run([str(exe)], env=env)
    reps = race_reports(out)
    if code is None:
        print("  races        timed out under syncsan")
        ok = False
    elif not reps:
        print("  races        none")
    else:
        print(f"  races        {len(reps)} racing pair{'s' if len(reps) > 1 else ''}; the first:")
        for rep in reps[:None if verbose else 1]:
            for l in rep:
                print("    " + l.strip())
        ok = False
    if verbose:
        print("\n".join("    | " + l for l in out.splitlines() if l.startswith("syncsan") and "RACE" not in l))

    for mode, label in (("1", "adversarial"), ("control", "control    ")):
        code, out = run([str(exe)], env=dict(env, SYNCSAN_ADVERSARIAL=mode))
        good, msg = verdict_line(code, out)
        print(f"  {label}  {msg}")
        ok &= good

    c, par = cost(regs), man["par"]
    if ok:
        score = "par" if c == par else (f"{c - par} over par" if c > par else f"{par - c} under par: tell us how")
        print(f"  cost         {c} sync op{'s' if c != 1 else ''} (par {par}): {score}")
    print("  PASS" if ok else "  FAIL")
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", type=int, help="puzzle numbers (default: all)")
    ap.add_argument("--dir", default="puzzles", help="puzzles (default) or solutions")
    ap.add_argument("--validate", action="store_true", help="every puzzle must fail as given and every solution pass")
    ap.add_argument("--manifest", action="store_true", help="rewrite puzzles/manifest.json (maintainers)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every race report and syncsan's summary")
    a = ap.parse_args()
    if a.manifest:
        return make_manifest()
    arch, so = gpu_arch(), syncsan_so()
    pick = lambda d: [p for p in puzzle_files(d) if not a.which or int(p.name[:2]) in a.which]
    if a.validate:
        rows = []
        for p in pick("puzzles"):
            given = grade(p, arch, so, a.verbose)
            solved = grade(ROOT / "solutions" / p.name, arch, so, a.verbose)
            rows.append((p.stem, given, solved, given == "FAIL" and solved == "PASS" or given == solved == "SKIP"))
        print("\npuzzle                  as given  solution  valid")
        for n, g, s, v in rows:
            print(f"{n:<24}{g:<10}{s:<10}{'yes' if v else 'NO'}")
        sys.exit(0 if all(r[3] for r in rows) else 1)
    res = [(p.stem, grade(p, arch, so, a.verbose)) for p in pick(a.dir)]
    print(f"\n{sum(r == 'PASS' for _, r in res)} of {len(res)} pass" +
          "".join(f", {n} skipped" for n in [sum(r == 'SKIP' for _, r in res)] if n))
    sys.exit(0 if all(r != "FAIL" for _, r in res) else 1)


if __name__ == "__main__":
    main()
