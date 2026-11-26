#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/scripts/sync_mesa_overlay.py" ] && [ -d "$SCRIPT_DIR/mesa" ] && command -v python3 >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/scripts/sync_mesa_overlay.py" --mode auto --quiet || true
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REMOVE=true
SLIM=false
ARCHIVE=false
TARGET_FILE="archive.zip"

ARGS=()
for arg in "$@"; do
    case "$arg" in
        -k|--keep)
            REMOVE=false
            ;;
        -s|--slim)
            SLIM=true
            ;;
        -a|--archive)
            SLIM=true
            ARCHIVE=true
            TARGET_FILE="archives/archive-$TIMESTAMP.zip"
            mkdir -p archives/
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

if [ "${#ARGS[@]}" -gt 0 ]; then
    TARGET_FILE="${ARGS[0]}"
fi
if $REMOVE; then
    rm -f "$TARGET_FILE"
fi

if command -v zip >/dev/null 2>&1; then
    FLAGS=(
        -9 -y -r "$TARGET_FILE" .
        -x ".cache/*" "build*/*" "archives/*" "archive.zip"
    )
    if $SLIM; then
        FLAGS+=(-x "*.so")
        FLAGS+=(-x "*.ko")
        FLAGS+=(-x "*.pyc")
        FLAGS+=(-x "*.zip")
        FLAGS+=(-x ".git/*")
        FLAGS+=(-x "predownloaded/*")
    fi
    zip "${FLAGS[@]}"
fi

ls -lh "$TARGET_FILE"
