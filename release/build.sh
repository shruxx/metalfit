#!/bin/sh
# Builds the release: MetalfitBar.app carrying its own Python, metalfit and llama-server, so installing it
# needs nothing else on the Mac - macOS ships Python 3.9 and metalfit needs 3.11.  Writes into dist/:
#   MetalfitBar-macos-arm64.zip   the app
#   install.sh                    curl ... | sh, puts the app into ~/Applications
#   SHA256SUMS                    which install.sh checks the zip against
#
#   LLAMA_SERVER  a llama-server built with Metal (required).  Build it for the oldest macOS it should run on,
#                 e.g. cmake -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DGGML_METAL_MACOSX_VERSION_MIN=14.0, and with
#                 GGML_METAL_EMBED_LIBRARY=ON and BUILD_SHARED_LIBS=OFF so it is one file.
#   LLAMA_LICENSE llama.cpp's LICENSE, shipped beside it (default: next to LLAMA_SERVER's source, else required)
#   PYTHON_HOME   a relocatable CPython >= 3.11 (default: uv's managed 3.12, from python-build-standalone)
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)"
die() { echo "release: $*" >&2; exit 1; }

[ -x "${LLAMA_SERVER:-}" ] || die "set LLAMA_SERVER to a llama-server built with Metal"
[ -f "${LLAMA_LICENSE:-}" ] || die "set LLAMA_LICENSE to llama.cpp's LICENSE file"
if [ -z "$PYTHON_HOME" ]; then
    command -v uv >/dev/null || die "no PYTHON_HOME and no uv to fetch one"
    uv python install -q 3.12
    # sys.base_prefix, not the path uv prints: that is a 3.12 alias whose bin/python links to an absolute path
    PYTHON_HOME="$("$(uv python find --managed-python 3.12)" -c 'import sys; print(sys.base_prefix)')"
fi
"$PYTHON_HOME/bin/python3" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "$PYTHON_HOME is older than 3.11"

# 1. the app itself, without a server command: it runs what it carries
METALFIT_BUNDLED=1 menubar/build.sh
APP="$ROOT/menubar/MetalfitBar.app"
RES="$APP/Contents/Resources"

# 2. Python, without what a server never imports (tests, Tk, IDLE, headers, pip): 120 MB down to ~45
ditto "$PYTHON_HOME" "$RES/python"
PYLIB="$(echo "$RES"/python/lib/python3.*)"
rm -rf "$RES/python/include" "$RES/python/share" "$RES"/python/lib/tcl* "$RES"/python/lib/tk* \
       "$RES"/python/lib/itcl* "$RES"/python/lib/thread* "$RES"/python/lib/libtcl* "$RES"/python/lib/libtk* \
       "$PYLIB/test" "$PYLIB/idlelib" "$PYLIB/tkinter" "$PYLIB/turtledemo" "$PYLIB/ensurepip" \
       "$PYLIB"/config-* "$PYLIB"/lib-dynload/_tkinter* "$PYLIB/site-packages"/*
find "$RES/python" -name __pycache__ -type d -prune -exec rm -rf {} +
for l in $(find "$RES/python" -type l); do
    case "$(readlink "$l")" in /*) die "$l links outside the app: $(readlink "$l")";; esac
done

# 3. metalfit, as a package in that Python's site-packages, and every .pyc made now: a Python that writes
#    __pycache__ into the app later breaks its signature
ditto metalfit "$PYLIB/site-packages/metalfit"
find "$PYLIB/site-packages/metalfit" -name __pycache__ -type d -prune -exec rm -rf {} +
"$RES/python/bin/python3" -m compileall -q -j 0 "$PYLIB" >/dev/null || true
"$RES/python/bin/python3" -m metalfit --help >/dev/null || die "the bundled metalfit does not start"

# 4. llama-server and the licences of everything shipped
mkdir -p "$RES/llama" "$RES/licenses"
cp "$LLAMA_SERVER" "$RES/llama/llama-server"
"$RES/llama/llama-server" --version >/dev/null 2>&1 || die "the bundled llama-server does not start"
cp "$LLAMA_LICENSE" "$RES/licenses/llama.cpp.txt"
if [ -f "$PYLIB/LICENSE.txt" ]; then
    cp "$PYLIB/LICENSE.txt" "$RES/licenses/python.txt"
else                                    # uv's builds leave it out: the same text from CPython's own tag
    PYVER="$("$RES/python/bin/python3" -c 'import platform; print(platform.python_version())')"
    curl -fsSL -o "$RES/licenses/python.txt" "https://raw.githubusercontent.com/python/cpython/v$PYVER/LICENSE" \
        || die "no Python licence to ship"
fi
cp LICENSE "$RES/licenses/metalfit.txt"

# 5. sign everything inside (ad hoc: there is no Developer ID) and pack
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
rm -rf dist && mkdir dist
ditto -c -k --keepParent "$APP" dist/MetalfitBar-macos-arm64.zip
cp release/install.sh dist/install.sh
(cd dist && shasum -a 256 MetalfitBar-macos-arm64.zip install.sh > SHA256SUMS)

echo "metalfit $VERSION: $(du -sh "$APP" | cut -f1) app, $(du -h dist/MetalfitBar-macos-arm64.zip | cut -f1) zipped"
echo "llama-server $("$RES/llama/llama-server" --version 2>&1 | sed -n 's/^version: //p'), $("$RES/python/bin/python3" -V)"
ls -l dist
