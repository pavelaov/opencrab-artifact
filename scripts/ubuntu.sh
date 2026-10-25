#!/usr/bin/env bash
# OpenCrab v2 on Ubuntu 22.04: bash scripts/ubuntu.sh <command> [run_campaign options]
#
#   setup   apt packages, CPython 3.12.10 in .venv, requirements.txt, the pinned GPT-2 snapshot
#   build   the C++ mirror with CMake and GCC; executables in cpp/build/bin
#   test    C++ tests (hardware and portable AES), then the Python suite
#   quick   pipeline check of every stage with the tiny model, into runs/quick
#   full    the paper campaign (every stage but the calibration), into runs/full
#   tables  the tables of results/ regenerated into runs/tables_check and compared with results/tables
#
# Only the apt step of setup needs sudo. The script finds the artifact root from
# its own path, so it runs from any directory. OPENCRAB2_PYTHON=<interpreter>
# replaces .venv/bin/python (it must be CPython 3.12 with requirements.txt).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY_VERSION=3.12.10
UV_VERSION=0.13.0
CHECK_ROOT="$ROOT/runs/tables_check"

usage() {
  echo "usage: bash scripts/ubuntu.sh <setup|build|test|quick|full|tables> [run_campaign options]" >&2
  exit 2
}

python_bin() {
  local py
  if [ -n "${OPENCRAB2_PYTHON:-}" ]; then
    py="$OPENCRAB2_PYTHON"
  elif [ -x "$ROOT/.venv/bin/python" ]; then
    py="$ROOT/.venv/bin/python"
  else
    echo "no interpreter: run 'bash scripts/ubuntu.sh setup' first or set OPENCRAB2_PYTHON" >&2
    return 1
  fi
  if ! "$py" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo "$py is not CPython 3.12" >&2
    return 1
  fi
  echo "$py"
}

loopback_no_proxy() {
  # loopback HTTP between the local processes must never go through a proxy
  export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
  export no_proxy="$NO_PROXY"
}

cmd_setup() {
  local sudo=sudo
  if [ "$(id -u)" -eq 0 ]; then
    sudo=""
  fi
  echo "[setup] apt: build-essential cmake (needs sudo)"
  $sudo apt-get update
  $sudo apt-get install -y build-essential cmake curl ca-certificates

  # CPython 3.12.10 (the measured interpreter) through uv, into .venv; no root needed.
  # Alternative with root, deadsnakes PPA (it carries a later 3.12 patch release):
  #   sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
  #   sudo apt-get install -y python3.12 python3.12-venv && python3.12 -m venv .venv
  local uv
  uv="$(command -v uv || true)"
  if [ -z "$uv" ]; then
    uv="$HOME/.local/bin/uv"
    if [ ! -x "$uv" ]; then
      echo "[setup] installing uv $UV_VERSION into $HOME/.local/bin"
      curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
    fi
  fi
  if [ ! -x "$ROOT/.venv/bin/python" ]; then
    "$uv" python install "$PY_VERSION"
    "$uv" venv --python "$PY_VERSION" --seed "$ROOT/.venv"
  fi
  local py
  py="$(OPENCRAB2_PYTHON="" python_bin)"
  "$py" -m pip install -r requirements.txt

  echo "[setup] GPT-2 snapshot (548 MB) into the Hugging Face cache"
  "$py" - <<'EOF'
import os
import sys
os.environ['HF_HUB_OFFLINE'] = '0'
sys.path.insert(0, 'python')
from huggingface_hub import snapshot_download
from opencrab2.highlevel import transformer as t
files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
snapshot_download(t.GPT2_REPO, revision=t.GPT2_REVISION, allow_patterns=files)
snap, rev = t.locate_snapshot(t.GPT2_REPO)
tok = t.load_gpt2_tokenizer()
print('[setup] GPT-2', t.GPT2_REPO, 'at', rev, 'in', snap, 'vocabulary', len(tok))
EOF
}

cmd_build() {
  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build cpp/build --parallel "$(nproc)"
}

cmd_test() {
  local py rc=0
  py="$(python_bin)"
  loopback_no_proxy
  if [ ! -f cpp/build/CTestTestfile.cmake ]; then
    echo "no C++ build in cpp/build: run 'bash scripts/ubuntu.sh build' first" >&2
    return 1
  fi
  ctest --test-dir cpp/build --output-on-failure || rc=1
  OPENCRAB2_AES=portable ctest --test-dir cpp/build --output-on-failure || rc=1
  "$py" -m pytest python/tests || rc=1
  return "$rc"
}

cmd_quick() {
  local py
  py="$(python_bin)"
  loopback_no_proxy
  "$py" scripts/run_campaign.py --quick "$@"
}

cmd_full() {
  # The declared final campaign: every stage but the calibration (the shipped fitted model
  # prices it), both session matrices with 3 workers; '--stage all' adds a fresh calibration.
  # Wall clock of the first campaign on the authors' host (Windows, 2026-10-11, diag4 at
  # 6 workers): primitives 50 s, cpp_bench 128 s, simulation 32 min (16 workers), diag4 2.8 h,
  # full12 1.2 h (3 workers), web 44 min, tables 2 s; calibration 4.6 h.
  local py
  py="$(python_bin)"
  loopback_no_proxy
  "$py" scripts/run_campaign.py "$@"
}

cmd_tables() {
  local py f name ok=0
  py="$(python_bin)"
  loopback_no_proxy
  shopt -s nullglob
  rm -rf "$CHECK_ROOT"
  mkdir -p "$CHECK_ROOT/raw" "$CHECK_ROOT/calibration"
  cp results/calibration/*.json "$CHECK_ROOT/calibration/"
  for f in results/raw/*.jsonl results/raw/*.jsonl.gz; do
    cp "$f" "$CHECK_ROOT/raw/"
  done
  if ! "$py" scripts/run_campaign.py --stage tables --out-root "$CHECK_ROOT"; then
    echo "[tables] make_tables reported errors (see $CHECK_ROOT/tables/manifest.json)"
    ok=1
  fi
  local shipped=(results/tables/*.tex)
  echo "[tables] results/tables against $CHECK_ROOT/tables"
  if [ "${#shipped[@]}" -eq 0 ]; then
    echo "  results/tables holds no tables: nothing to compare"
    return 1
  fi
  for f in "${shipped[@]}"; do
    name="$(basename "$f")"
    if [ ! -f "$CHECK_ROOT/tables/$name" ]; then
      echo "  missing  $name"
      ok=1
    elif cmp -s "$f" "$CHECK_ROOT/tables/$name"; then
      echo "  equal    $name"
    else
      echo "  differs  $name"
      ok=1
    fi
  done
  for f in "$CHECK_ROOT"/tables/*.tex; do
    name="$(basename "$f")"
    if [ ! -f "results/tables/$name" ]; then
      echo "  new      $name"
      ok=1
    fi
  done
  if [ "$ok" -eq 0 ]; then
    echo "[tables] every table equals results/tables"
  else
    echo "[tables] the regenerated tables differ from results/tables"
  fi
  return "$ok"
}

[ "$#" -ge 1 ] || usage
cmd="$1"
shift
case "$cmd" in
  setup) cmd_setup ;;
  build) cmd_build ;;
  test) cmd_test ;;
  quick) cmd_quick "$@" ;;
  full) cmd_full "$@" ;;
  tables) cmd_tables ;;
  *) usage ;;
esac
