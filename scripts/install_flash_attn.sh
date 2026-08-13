#!/usr/bin/env bash
# install_flash_attn.sh — Install a *prebuilt* flash-attn 2 wheel.
#
# Never build flash-attn from source on a rented GPU: `pip install flash-attn`
# compiles CUDA kernels for 30–90 minutes and often OOMs the box. Dao-AILab
# publishes wheels for every (CUDA, torch, python, C++ ABI) combination, so we
# resolve the matching one and download it directly.
#
# Wheel names look like:
#   flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
#                          ^cuda  ^torch  ^abi        ^python
#
# Usage:
#   bash scripts/install_flash_attn.sh            # into the active venv
#   FA_VERSION=2.8.3.post1 bash scripts/install_flash_attn.sh
#
# Exits 0 even when it cannot install: flash-attn is a speedup, not a
# requirement — main.py falls back to sdpa on its own.

set -uo pipefail

# Fallback when the release API is unreachable. v2.8.3 is the widest FA2 build
# matrix as of Aug 2026: cu12torch2.4–2.9, cu13torch2.9–2.10, cp39–cp313.
# (v2.8.3.post1 is newer but ships fewer torch variants.)
FA_FALLBACK_VERSION="${FA_VERSION:-2.8.3}"
RELEASES_API="https://api.github.com/repos/Dao-AILab/flash-attention/releases"
DOWNLOAD_BASE="https://github.com/Dao-AILab/flash-attention/releases/download"

PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

say() { echo "[flash-attn] $*"; }

# Already there? Nothing to do.
if "$PY" -c "import flash_attn" 2>/dev/null; then
    say "already installed: $("$PY" -c 'import flash_attn; print(flash_attn.__version__)')"
    exit 0
fi

# Probe the environment we have to match exactly.
read -r TORCH_MM CUDA_MAJOR ABI PYTAG SM <<<"$("$PY" - <<'PYEOF'
import sys

try:
    import torch
except ImportError:
    print("none none none none none")
    raise SystemExit

torch_mm = ".".join(torch.__version__.split(".")[:2])
cuda_major = (torch.version.cuda or "none").split(".")[0]
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
pytag = f"cp{sys.version_info.major}{sys.version_info.minor}"
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"{major}{minor}"
else:
    sm = "none"
print(torch_mm, cuda_major, abi, pytag, sm)
PYEOF
)"

if [ "$TORCH_MM" = "none" ]; then
    say "torch is not importable — skipping."
    exit 0
fi
if [ "$CUDA_MAJOR" = "none" ]; then
    say "this torch build has no CUDA (CPU wheel) — skipping; sdpa will be used."
    exit 0
fi
if [ "$SM" = "none" ]; then
    say "no GPU visible — skipping."
    exit 0
fi
if [ "$SM" -lt 80 ] 2>/dev/null; then
    say "GPU is sm_$SM; flash-attn 2 needs sm_80+ (Ampere or newer) — skipping."
    exit 0
fi

# The same wheel is published for x86_64 and aarch64 — match the platform too,
# or an ARM wheel can win the search on an Intel box.
case "$(uname -m)" in
    aarch64|arm64) PLAT="linux_aarch64" ;;
    *)             PLAT="linux_x86_64" ;;
esac

TAGS="cu${CUDA_MAJOR}torch${TORCH_MM}cxx11abi${ABI}-${PYTAG}-${PYTAG}-${PLAT}.whl"
say "matching torch $TORCH_MM / cu$CUDA_MAJOR / $PYTAG / cxx11abi=$ABI / $PLAT / sm_$SM"

# Ask GitHub for the newest release carrying a wheel for this combination.
# The release feed is mostly flash-attn 4 betas, so fetch enough pages back to
# reach the flash-attn 2 tags.
RELEASES_JSON=""
if command -v curl >/dev/null 2>&1; then
    RELEASES_JSON="$(curl -fsSL "$RELEASES_API?per_page=40" 2>/dev/null || true)"
fi

# Kept in a variable, not a heredoc: a heredoc on the python call would take
# over stdin and the piped JSON would never arrive.
read -r -d '' PY_RESOLVER <<'PYEOF' || true
import json
import sys

tags, mode = sys.argv[1], sys.argv[2]
try:
    releases = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    raise SystemExit

assets = [
    asset
    for release in releases  # newest first
    for asset in release.get("assets", [])
    if asset.get("name", "").startswith("flash_attn-2.")
]

if mode == "url":
    for asset in assets:
        if tags in asset["name"]:
            print(asset["browser_download_url"])
            break
else:  # what this feed *does* offer, for the failure message
    combos = {a["name"].split("+", 1)[1].split("cxx11abi")[0] for a in assets if "+" in a["name"]}
    print(" ".join(sorted(combos)))
PYEOF

resolve_wheel() {  # $1 = "url" | "combos"
    printf '%s' "$RELEASES_JSON" | "$PY" -c "$PY_RESOLVER" "$TAGS" "$1"
}

WHEEL_URL=""
[ -n "$RELEASES_JSON" ] && WHEEL_URL="$(resolve_wheel url)"

if [ -z "$WHEEL_URL" ]; then
    WHEEL_URL="$DOWNLOAD_BASE/v${FA_FALLBACK_VERSION}/flash_attn-${FA_FALLBACK_VERSION}+${TAGS}"
    if [ -n "$RELEASES_JSON" ]; then
        say "no published wheel matches this environment — trying pinned v$FA_FALLBACK_VERSION"
        COMBOS="$(resolve_wheel combos)"
        [ -n "$COMBOS" ] && say "  published builds: $COMBOS"
    else
        say "release API unavailable — trying pinned v$FA_FALLBACK_VERSION"
    fi
fi

say "installing $(basename "$WHEEL_URL")"

# --no-deps keeps pip from touching the preinstalled torch (a resolver run here
# can silently swap in a CPU build). einops is flash-attn's only other need.
if command -v uv >/dev/null 2>&1; then
    INSTALL="uv pip install"
else
    INSTALL="$PY -m pip install"
fi

if ! $INSTALL --no-deps "$WHEEL_URL"; then
    say "WARNING: install failed — continuing without flash-attn (sdpa fallback)."
    say "  This torch ($TORCH_MM) may be newer than the newest flash-attn 2 build."
    say "  Pin a supported torch, or browse:"
    say "  https://github.com/Dao-AILab/flash-attention/releases"
    exit 0
fi
$INSTALL einops >/dev/null 2>&1 || true

if "$PY" -c "import flash_attn" 2>/dev/null; then
    say "OK — flash_attn $("$PY" -c 'import flash_attn; print(flash_attn.__version__)') importable."
else
    say "WARNING: wheel installed but 'import flash_attn' fails — sdpa fallback will be used."
fi
exit 0
