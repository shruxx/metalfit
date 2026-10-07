#!/bin/sh
# Builds MetalfitBar.app next to this script: a menu bar item showing which model metalfit has loaded.
# Needs only the Xcode command line tools (swiftc).  Copy the app to ~/Applications; its menu's "Open at login"
# starts it at every login.  release/build.sh builds on this for the self-contained release.
set -e
cd "$(dirname "$0")"
APP=MetalfitBar.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -parse-as-library -target arm64-apple-macos13.0 -o "$APP/Contents/MacOS/MetalfitBar" MetalfitBar.swift Icon.swift

# the app icon, drawn by the same code as the menu bar item
TMP="$(mktemp -d)"
swiftc -O -parse-as-library -o "$TMP/makeicon" makeicon.swift AppIcon.swift Icon.swift
"$TMP/makeicon" "$TMP/AppIcon.iconset"
iconutil -c icns -o "$APP/Contents/Resources/AppIcon.icns" "$TMP/AppIcon.iconset"
rm -rf "$TMP"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' ../pyproject.toml)"
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>MetalfitBar</string>
    <key>CFBundleIdentifier</key><string>io.github.shruxx.metalfit.bar</string>
    <key>CFBundleExecutable</key><string>MetalfitBar</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSUIElement</key><true/>
    <key>NSAppTransportSecurity</key>
    <dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict>
</plist>
EOF

# The command "Start server" runs, found now so the app works from /Applications too:
#   METALFIT_BIN           metalfit itself, else the one on PATH, else this checkout's .venv
#   METALFIT_MODELS        the model folder (default ~/models)
#   METALFIT_LLAMA_SERVER  llama-server, else the one on PATH, else metalfit looks for it itself
#   METALFIT_ARGS          anything else for `metalfit serve`, e.g. "--preload Q3_K_XL -c 65536"
# Change it later without rebuilding: defaults write io.github.shruxx.metalfit.bar command -array ...
#
# None of these should be under ~/Documents, ~/Desktop or ~/Downloads.  macOS asks the app for permission to
# read those folders, and until someone answers the server hangs in its first open() - Python reading its
# pyvenv.cfg - and the ad-hoc signature changes with every build, so it asks again.  `uv tool install .`
# puts metalfit in ~/.local/bin, which is not protected.
#
# METALFIT_BUNDLED=1 writes no command: release/build.sh puts Python, metalfit and llama-server into the app,
# and the app runs those.
BIN="${METALFIT_BIN:-$(command -v metalfit || true)}"
[ -n "$BIN" ] || { [ -x "$(cd .. && pwd)/.venv/bin/metalfit" ] && BIN="$(cd .. && pwd)/.venv/bin/metalfit"; }
if [ -n "$METALFIT_BUNDLED" ]; then
    :
elif [ -z "$BIN" ]; then
    echo "warning: no metalfit found (METALFIT_BIN, PATH or .venv); 'Start server' will say so" >&2
else
    LLAMA="${METALFIT_LLAMA_SERVER:-$(command -v llama-server || true)}"
    MODELS="${METALFIT_MODELS:-$HOME/models}"
    for p in "$BIN" "$LLAMA" "$MODELS"; do
        case "$p" in "$HOME"/Documents/*|"$HOME"/Desktop/*|"$HOME"/Downloads/*)
            echo "warning: $p is in a folder macOS protects; the server may hang until access is granted" >&2;;
        esac
    done
    # shellcheck disable=SC2086
    set -- "$BIN" serve --models "$MODELS" ${LLAMA:+--llama-server "$LLAMA"} $METALFIT_ARGS
    plutil -insert MetalfitCommand -json "$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")" \
        "$APP/Contents/Info.plist"
    echo "server command: $*"
fi

codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "built $(pwd)/$APP"
