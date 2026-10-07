#!/bin/sh
# Installs metalfit: MetalfitBar.app, which carries its own Python and llama-server, into ~/Applications.
#
#   curl -fsSL https://github.com/shruxx/metalfit/releases/latest/download/install.sh | sh
#
# Run again to update.  METALFIT_VERSION=0.2.0 picks a release, METALFIT_APPS another folder than
# ~/Applications, METALFIT_URL another place to download from (a mirror, or a local test of dist/).
# Downloaded with curl the app carries no quarantine flag, so macOS opens it although it is signed ad hoc
# rather than with a Developer ID; a zip downloaded in a browser has to be opened once with a right-click >
# Open instead.
set -e
REPO=shruxx/metalfit
ZIP=MetalfitBar-macos-arm64.zip
say() { printf '%s\n' "metalfit: $*"; }
die() { say "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "this needs a Mac with Apple Silicon"
[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 14 ] || die "this needs macOS 14 or newer"

if [ -n "$METALFIT_URL" ]; then
    URL="$METALFIT_URL"
elif [ -n "$METALFIT_VERSION" ]; then
    URL="https://github.com/$REPO/releases/download/v$METALFIT_VERSION"
else
    URL="https://github.com/$REPO/releases/latest/download"
fi
DEST="${METALFIT_APPS:-$HOME/Applications}"
APP="$DEST/MetalfitBar.app"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
say "downloading from $URL"
curl -fL --progress-bar -o "$TMP/$ZIP" "$URL/$ZIP"
curl -fsSL -o "$TMP/SHA256SUMS" "$URL/SHA256SUMS"
want="$(awk -v f="$ZIP" '$2 == f {print $1}' "$TMP/SHA256SUMS")"
have="$(shasum -a 256 "$TMP/$ZIP" | cut -d' ' -f1)"
[ -n "$want" ] && [ "$want" = "$have" ] || die "checksum mismatch, nothing installed"

# an update: stop the app and the server it runs from inside the old copy, whose files are about to go
if [ -d "$APP" ]; then
    say "replacing $APP"
    pkill -x MetalfitBar 2>/dev/null || true
    pkill -TERM -f "$APP/Contents/Resources/python" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        pgrep -f "$APP/Contents/Resources/python" >/dev/null || break
        sleep 1                                   # metalfit unloads the model before it exits
    done
    rm -rf "$APP"
fi
mkdir -p "$DEST"
ditto -x -k "$TMP/$ZIP" "$DEST"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true

mkdir -p "$HOME/models"

# `metalfit` in the terminal too, unless one is there already (a uv or pip install stays in charge)
if ! command -v metalfit >/dev/null 2>&1; then
    mkdir -p "$HOME/.local/bin"
    cat > "$HOME/.local/bin/metalfit" <<EOF
#!/bin/sh
# metalfit from MetalfitBar.app (written by its install.sh)
PYTHONDONTWRITEBYTECODE=1 exec "$APP/Contents/Resources/python/bin/python3" -m metalfit "\$@"
EOF
    chmod +x "$HOME/.local/bin/metalfit"
    case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) say "add ~/.local/bin to PATH to use metalfit in a terminal";; esac
fi

# the first install starts the server with the app; a choice made in its menu later is kept
defaults read io.github.shruxx.metalfit.bar autostart >/dev/null 2>&1 \
    || defaults write io.github.shruxx.metalfit.bar autostart -bool true

open "$APP"
say "installed $(defaults read "$APP/Contents/Info" CFBundleShortVersionString 2>/dev/null) to $APP"
say "put .gguf models into ~/models; they appear in the menu bar item's menu"
say "to start it at every login: its menu > Open at login"
