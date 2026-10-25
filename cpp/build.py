#!/usr/bin/env python3
"""OpenCrab v2 C++17 mirror: build driver for a host without a system compiler.

The C++ tree has two build drivers. They compile the same sources with the same
flags (-std=c++17 -O2 -ffp-contract=off) and put the executables in cpp/build/bin,
where the Python tests and scripts look for them:

  * CMakeLists.txt with the system compiler, the driver on Ubuntu 22.04
    (GCC 11.4, CMake 3.22):
        cmake -S cpp -B cpp/build && cmake --build cpp/build --parallel
  * this script, the driver on Windows, where there is neither a system compiler
    nor CMake. It runs the clang that the Python ziglang package ships
    (pip install ziglang) with the interpreter that runs the script:
        python cpp/build.py        runs: python -m ziglang c++ -std=c++17 -O2 ...
    It also works on Linux with that package, but builds with clang and libc++
    there, not with GCC.

-ffp-contract=off keeps a multiplication and an addition two separate IEEE 754
operations. Without it clang fuses them into one fused multiply-add where the
CPU has that instruction (this driver compiles for the host CPU), a GCC build
for baseline x86-64 does not, and the cost-model predictions in the
cross-language vector files are compared exactly; with the flag every build
produces the same doubles.

Layout (relative to this file's directory, cpp/):
    include/opencrab2/<layer>/*.hpp   public headers (layers: lowlevel, highlevel, platform)
    src/<layer>/*.cpp                 library sources, compiled once into build/obj/<layer>/*.o
    apps/*.cpp                        one executable per file -> build/bin/<name>[.exe]
    tests/*.cpp                       one executable per file -> build/bin/<name>[.exe]

Every executable links against the object files of all three layers (the layers
are small; linking the full object set keeps one build rule for every target and
makes duplicate symbol definitions visible instead of hiding them in archives).
On Windows every target links -lws2_32 (winsock2); on other hosts -lpthread.

Usage:
    python build.py                      build everything (release, -O2)
    python build.py tests                the single test target: build every test executable,
                                         run each one and print ONE summary line
                                         "[tests] SUMMARY: ..." (exit 1 on any failure)
    python build.py --debug              -O0 -g, defines OPENCRAB2_DEBUG
    python build.py --jobs 8             thread pool size (default: os.cpu_count())
    python build.py --targets test_lowlevel opencrab2_dealer
    python build.py --run-tests          build the requested targets, then run the test
                                         executables among them (same summary line)
    python build.py --build-dir DIR      objects and executables under DIR instead of build/
                                         (private trees for concurrent builds)
    python build.py --keep-going         link what can be linked when a library source fails
                                         to compile (the exit status is still 1)
    python build.py --clean              remove the build directory
    python build.py --zig-target x86_64-linux-gnu.2.35    cross compile (compile + link only)
    python build.py --python PATH        another interpreter that has the ziglang package
                                         (default: the one running this script)
    python build.py --warnings           add -Wall -Wextra

Source manifest. Library sources and targets are discovered by globbing, so a
new file under src/<layer>/, apps/ or tests/ is built without editing this
script. EXPECTED_SOURCES additionally names files the tree is expected to
carry; a file of that list that does not exist is reported as
"SKIP <path>: source file missing" and the build goes on. A target
whose own source or one of its TARGET_REQUIRES sources is missing is skipped
the same way instead of failing the whole build; a requested target name that
is neither on disk nor in the manifest is an error (BUILD_UNKNOWN_TARGET).

Hardware AES. src/lowlevel/aes128.cpp carries an AES-NI path next to the portable one.
Its functions are compiled with a per-function target attribute, so no -maes flag is
passed here (and none in CMakeLists.txt); the path is selected at run time by CPUID.
OPENCRAB2_AES=portable in the environment forces the portable path for a whole run,
which is how the test executables are run a second time on that path:
    OPENCRAB2_AES=portable build/bin/test_lowlevel
    OPENCRAB2_AES=portable build/bin/test_vectors tests/vectors
-DOPENCRAB2_AES_NO_HARDWARE removes the hardware path at compile time.

Exit status: 0 when every requested target that has its sources built (and,
with the tests target or --run-tests, every test passed); 1 otherwise. The
script prints every failing compiler invocation verbatim. Incremental: an
object is rebuilt when its source or any header under include/ is newer than
the object, or when the flag set changed. A command that fails with a Zig
cache collision (two concurrent zig processes) is retried up to three times.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
INCLUDE = HERE / "include"
SRC = HERE / "src"
APPS = HERE / "apps"
TESTS = HERE / "tests"
BUILD = HERE / "build"
# Host release objects and executables live in build/obj and build/bin; a debug
# build uses build/debug/{obj,bin} and a cross build build/<target>/{obj,bin},
# so configurations never overwrite each other's object files. --build-dir
# replaces build/ by another root with the same substructure.
OBJ = BUILD / "obj"
BIN = BUILD / "bin"
LAYERS = ("lowlevel", "highlevel", "platform")

# Sources the tree is expected to carry in addition to whatever the globs find.
# A missing one is skipped with a notice (see the module docstring).
EXPECTED_SOURCES = (
    "src/highlevel/bridge.cpp",
    "tests/test_bridge.cpp",
    "tests/test_ledger.cpp",
)
# Known target names (apps and tests); tells a skipped target from a typo.
KNOWN_TARGETS = (
    "opencrab2_bench", "opencrab2_dealer", "opencrab2_party", "opencrab2_server", "opencrab2_sidecar",
    "test_bridge", "test_decision", "test_engine", "test_ledger", "test_lowlevel", "test_platform",
    "test_toy_bfv", "test_two_party", "test_vectors",
)
# Library sources a target cannot link without (beyond its own main file).
TARGET_REQUIRES = {
    "test_bridge": ("src/highlevel/bridge.cpp",),
}
# Extra command line arguments of a test executable.
TEST_ARGS = {
    "test_vectors": lambda: [str(HERE / "tests" / "vectors")],
}
# Compiler output that marks a transient Zig cache collision (the command is retried).
TRANSIENT_MARKERS = ("AccessDenied", "FileNotFound", "PermissionDenied", "sharing violation", "unable to build",
                     "failed to build", "CacheCheckFailed", "FileBusy", "Unexpected")

def find_python(explicit: Optional[str]) -> str:
    """The interpreter that carries the ziglang package: --python when given, else the running one."""
    candidate = explicit or sys.executable
    if candidate and Path(candidate).exists():
        try:
            r = subprocess.run([candidate, "-m", "ziglang", "version"], capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            pass
    raise SystemExit(f"BUILD_TOOLCHAIN_MISSING: {candidate} has no ziglang package (pip install ziglang, or pass --python); "
                     "with a system compiler build through CMakeLists.txt instead")


def exe_suffix(zig_target: Optional[str]) -> str:
    if zig_target:
        return ".exe" if "windows" in zig_target else ""
    return ".exe" if os.name == "nt" else ""


def is_windows_target(zig_target: Optional[str]) -> bool:
    if zig_target:
        return "windows" in zig_target
    return os.name == "nt"


def newest_header_mtime() -> float:
    newest = 0.0
    for p in INCLUDE.rglob("*.hpp"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue  # a header replaced by a concurrent editor between the scan and the stat
    return newest


def flags_hash(flags: List[str]) -> str:
    return hashlib.sha256(" ".join(flags).encode()).hexdigest()[:16]


def tree_hash() -> str:
    """Content hash of the C++ tree: the code_commit of every evidence record an app writes.

    sha256 over one line per file, "<sha256 of the file bytes>  <path relative to cpp/>"
    followed by a line feed, for every include/**/*.hpp, src/**/*.cpp and apps/*.cpp in
    sorted path order (forward slashes). CMakeLists.txt computes the same value with
    file(SHA256) and string(SHA256), so both build systems stamp the same identity for
    the same sources.
    """
    files = list(INCLUDE.rglob("*.hpp")) + list(SRC.rglob("*.cpp")) + list(APPS.glob("*.cpp"))
    lines = []
    for p in files:
        try:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            continue  # replaced by a concurrent editor during the scan
        lines.append((p.relative_to(HERE).as_posix(), digest))
    lines.sort()
    text = "".join(digest + "  " + rel + chr(10) for rel, digest in lines)
    return "cpp-tree-" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_transient(output: str) -> bool:
    """True when a failed compiler run looks like a cache collision, not a diagnostic of our sources."""
    if ".cpp:" in output or ".hpp:" in output:
        return False  # a real diagnostic names a source location
    return any(m in output for m in TRANSIENT_MARKERS)


class Build:
    def __init__(self, args: argparse.Namespace) -> None:
        self.python = find_python(args.python)
        self.zig_target = args.zig_target
        self.debug = args.debug
        self.verbose = args.verbose
        self.keep_going = bool(args.keep_going)
        self.jobs = args.jobs or (os.cpu_count() or 4)
        self.cxx = [self.python, "-m", "ziglang", "c++"]
        # -ffp-contract=off: no fused multiply-add, the same doubles as a GCC build (module docstring)
        self.common = ["-std=c++17", "-ffp-contract=off", "-I" + str(INCLUDE)]
        if self.debug:
            self.common += ["-O0", "-g", "-DOPENCRAB2_DEBUG"]
        else:
            self.common += ["-O2"]
        if args.warnings:
            self.common += ["-Wall", "-Wextra"]
        if self.zig_target:
            self.common += ["-target", self.zig_target]
        self.link_libs = ["-lws2_32"] if is_windows_target(self.zig_target) else ["-lpthread"]
        self.suffix = exe_suffix(self.zig_target)
        self.header_mtime = newest_header_mtime()
        self.skipped: List[str] = []
        config_dir = Path(args.build_dir).resolve() if args.build_dir else BUILD
        if self.zig_target:
            config_dir = config_dir / self.zig_target.replace("/", "_")
        if self.debug:
            config_dir = config_dir / "debug"
        self.obj_dir = config_dir / "obj"
        self.bin_dir = config_dir / "bin"
        self.stamp_path = config_dir / "flags.json"
        self.flag_id = flags_hash(self.common + [self.zig_target or "host"])
        # Stamped into the main translation unit of every app and test only (never into the
        # library objects, which would go stale): see platform/evidence.hpp.
        self.code_commit = tree_hash()

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def show(path: Path) -> str:
        """Path relative to cpp/ when it lies inside it, absolute otherwise (private build trees)."""
        try:
            return path.relative_to(HERE).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def run_tool(cmd: List[str]) -> Tuple[bool, str]:
        """Run one compiler command; retry a transient Zig cache collision up to three times."""
        out = ""
        for attempt in range(3):
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            out = (r.stdout or "") + (r.stderr or "")
            if r.returncode == 0:
                return True, out
            if not is_transient(out):
                break
            time.sleep(1.0 + attempt)
        return False, out

    # ---- compilation -------------------------------------------------------
    def object_path(self, source: Path) -> Path:
        rel = source.relative_to(HERE).with_suffix(".o")
        return self.obj_dir / rel

    def needs_rebuild(self, source: Path, obj: Path, previous_flag_id: str) -> bool:
        if not obj.exists():
            return True
        if previous_flag_id != self.flag_id:
            return True
        om = obj.stat().st_mtime
        return source.stat().st_mtime > om or self.header_mtime > om

    def compile_one(self, source: Path) -> Tuple[Path, bool, str, float]:
        obj = self.object_path(source)
        obj.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.cxx + self.common + ["-c", str(source), "-o", str(obj)]
        t0 = time.time()
        good, out = self.run_tool(cmd)
        return obj, good, out, time.time() - t0

    def link_one(self, main_source: Path, objects: List[Path]) -> Tuple[Path, bool, str, float]:
        exe = self.bin_dir / (main_source.stem + self.suffix)
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        cmd = (self.cxx + self.common + ["-DOPENCRAB2_CODE_COMMIT=" + self.code_commit, str(main_source)] +
               [str(o) for o in objects] + ["-o", str(exe)] + self.link_libs)
        t0 = time.time()
        good, out = self.run_tool(cmd)
        return exe, good, out, time.time() - t0

    # ---- driver ------------------------------------------------------------
    def run(self, targets: Optional[List[str]]) -> Tuple[bool, List[Path]]:
        self.obj_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        previous = ""
        if self.stamp_path.exists():
            try:
                previous = json.loads(self.stamp_path.read_text()).get("flag_id", "")
            except (OSError, ValueError):
                previous = ""

        lib_sources: List[Path] = []
        for layer in LAYERS:
            lib_sources += sorted((SRC / layer).glob("*.cpp"))
        mains = sorted(APPS.glob("*.cpp")) + sorted(TESTS.glob("*.cpp"))
        # Expected files that are not there (yet): report and go on.
        for rel in EXPECTED_SOURCES:
            if not (HERE / rel).exists():
                self.skipped.append(rel)
                print(f"  SKIP {rel}: source file missing")
        on_disk = {m.stem for m in mains}
        if targets:
            wanted = list(dict.fromkeys(targets))
            unknown = [t for t in wanted if t not in on_disk and t not in KNOWN_TARGETS]
            if unknown:
                print("BUILD_UNKNOWN_TARGET: " + ", ".join(sorted(unknown)))
                return False, []
            for t in wanted:
                if t not in on_disk and ("tests/" + t + ".cpp") not in self.skipped and ("apps/" + t + ".cpp") not in self.skipped:
                    self.skipped.append(t)
                    print(f"  SKIP target {t}: source file missing")
            mains = [m for m in mains if m.stem in set(wanted)]
        # A target that needs a library source which is missing is skipped, not failed.
        kept = []
        for m in mains:
            missing = [rel for rel in TARGET_REQUIRES.get(m.stem, ()) if not (HERE / rel).exists()]
            if missing:
                self.skipped.append(m.stem)
                print(f"  SKIP target {m.stem}: requires {', '.join(missing)} (missing)")
            else:
                kept.append(m)
        mains = kept

        to_compile = [s for s in lib_sources if self.needs_rebuild(s, self.object_path(s), previous)]
        print(f"[build] python={self.python}")
        print(f"[build] flags={' '.join(self.common)} libs={' '.join(self.link_libs)} jobs={self.jobs}")
        print(f"[build] objects: {self.show(self.obj_dir)}  executables: {self.show(self.bin_dir)}")
        print(f"[build] code_commit={self.code_commit}")
        print(f"[build] library sources: {len(lib_sources)} ({len(to_compile)} to compile), targets: {len(mains)}")

        ok = True
        failed_sources = set()
        t_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futures = {pool.submit(self.compile_one, s): s for s in to_compile}
            for fut in concurrent.futures.as_completed(futures):
                src = futures[fut]
                obj, good, out, dt = fut.result()
                rel = src.relative_to(HERE).as_posix()
                if good:
                    print(f"  CC  {rel}  ({dt:.1f}s)")
                    if out.strip() and self.verbose:
                        print(out)
                else:
                    ok = False
                    failed_sources.add(src)
                    print(f"  FAIL {rel}")
                    print(out)
        if not ok and not self.keep_going:
            print("[build] compilation failed; not linking (--keep-going links the remaining objects)")
            return False, []
        if ok:
            self.stamp_path.write_text(json.dumps({"flag_id": self.flag_id}))

        objects = [self.object_path(s) for s in lib_sources if s not in failed_sources]
        built: List[Path] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futures = {pool.submit(self.link_one, m, objects): m for m in mains}
            for fut in concurrent.futures.as_completed(futures):
                main = futures[fut]
                exe, good, out, dt = fut.result()
                rel = main.relative_to(HERE).as_posix()
                if good:
                    print(f"  LD  {rel} -> {self.show(exe)}  ({dt:.1f}s)")
                    if out.strip() and self.verbose:
                        print(out)
                    built.append(exe)
                else:
                    ok = False
                    print(f"  FAIL {rel}")
                    print(out)
        print(f"[build] {'ok' if ok else 'FAILED'} in {time.time() - t_start:.1f}s")
        return ok, sorted(built)


def run_tests(executables: List[Path], skipped: Optional[List[str]] = None, not_built: Optional[List[str]] = None) -> bool:
    """Run every test executable (each prints 'PASS <n> checks' and exits 0 on success).

    Prints one line per executable and exactly ONE final line
        [tests] SUMMARY: <passed>/<total> executables passed, <failed> failed, <skipped> skipped, <n> checks, <t> s
    where total counts the executables that were run plus the test targets that
    failed to build (those count as failed) and skipped counts the test targets
    whose source file is missing.
    """
    all_ok = True
    total_checks = 0
    passed = 0
    failed: List[str] = list(not_built or [])
    t_all = time.time()
    tests = [exe for exe in executables if exe.stem.startswith("test_")]
    skipped_tests = sorted({Path(s).stem for s in (skipped or []) if Path(s).stem.startswith("test_")})
    for exe in tests:
        make_args = TEST_ARGS.get(exe.stem)
        args = make_args() if make_args else []
        t0 = time.time()
        try:
            r = subprocess.run([str(exe)] + args, capture_output=True, text=True, cwd=str(HERE.parent), timeout=3600,
                               encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT {exe.name}")
            failed.append(exe.stem)
            continue
        dt = time.time() - t0
        out = (r.stdout or "") + (r.stderr or "")
        pass_line = ""
        for line in out.splitlines():
            if re.fullmatch(r"PASS \d+ checks", line.strip()):
                pass_line = line.strip()
        if r.returncode == 0 and pass_line:
            try:
                total_checks += int(pass_line.split()[1])
            except (IndexError, ValueError):
                pass
            passed += 1
            print(f"  OK   {exe.name}: {pass_line}  ({dt:.1f}s)")
        else:
            failed.append(exe.stem)
            print(f"  FAIL {exe.name} (exit {r.returncode}, {dt:.1f}s)")
            print(out[-4000:])
    total = len(tests) + len(not_built or [])
    if failed or total == 0:
        all_ok = False
    detail = ""
    if failed:
        detail += " failed=[" + ",".join(failed) + "]"
    if skipped_tests:
        detail += " skipped=[" + ",".join(skipped_tests) + "]"
    print(f"[tests] SUMMARY: {passed}/{total} executables passed, {len(failed)} failed, {len(skipped_tests)} skipped, "
          f"{total_checks} checks, {time.time() - t_all:.1f} s{detail}")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("goal", nargs="*", help="'tests' (build and run every test executable, one summary line) "
                                            "or target names (same as --targets)")
    ap.add_argument("--debug", action="store_true", help="-O0 -g build")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--targets", nargs="*", help="only these app/test names ('tests' selects the test target)")
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--build-dir", default=None, help="build tree (default: cpp/build)")
    ap.add_argument("--keep-going", action="store_true", help="link the remaining objects after a compile failure")
    ap.add_argument("--zig-target", default=None, help="e.g. x86_64-linux-gnu.2.35")
    ap.add_argument("--python", default=None, help="interpreter with the ziglang package (default: this one)")
    ap.add_argument("--warnings", action="store_true", help="add -Wall -Wextra")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    build_root = Path(args.build_dir).resolve() if args.build_dir else BUILD
    requested = list(args.targets or []) + list(args.goal or [])
    tests_goal = "tests" in requested
    requested = [t for t in requested if t != "tests"]
    if args.clean:
        if build_root.exists():
            shutil.rmtree(build_root)
            print("[build] removed " + str(build_root))
        if not args.run_tests and not requested and not tests_goal:
            return 0

    targets: Optional[List[str]] = requested or None
    if tests_goal:
        # the single test target: every tests/*.cpp plus the tests the manifest expects
        names = {p.stem for p in TESTS.glob("*.cpp")} | {t for t in KNOWN_TARGETS if t.startswith("test_")}
        targets = sorted(names | set(requested))
    b = Build(args)
    ok, built = b.run(targets)
    if args.run_tests or tests_goal:
        if args.zig_target and not (is_windows_target(args.zig_target) == (os.name == "nt")):
            print("[tests] SUMMARY: not run, cross-compiled executables cannot run on this host")
            return 0 if ok else 1
        wanted_tests = {t for t in (targets or [p.stem for p in TESTS.glob("*.cpp")]) if t.startswith("test_")}
        skipped_names = {Path(s).stem for s in b.skipped}
        not_built = sorted(wanted_tests - {e.stem for e in built} - skipped_names)
        if not run_tests(built, b.skipped, not_built):
            return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
