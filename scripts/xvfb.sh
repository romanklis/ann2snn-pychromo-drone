#!/usr/bin/env bash
# Run a command under a temporary Xvfb display.
#
# xvfb-run hangs in this image (the base is a ROS/Ubuntu image with an unusual
# shell setup), so start Xvfb directly, wait for its socket, run the command,
# and always tear the server down.
#
# Usage: scripts/xvfb.sh <command> [args...]
set -euo pipefail

display="${XVFB_DISPLAY:-:99}"
screen="${XVFB_SCREEN:-1280x720x24}"
log="${XVFB_LOG:-/tmp/xvfb.log}"

Xvfb "$display" -screen 0 "$screen" -nolisten tcp >"$log" 2>&1 &
xvfb_pid=$!

cleanup() {
    kill "$xvfb_pid" 2>/dev/null || true
    wait "$xvfb_pid" 2>/dev/null || true
}
trap cleanup EXIT

socket="/tmp/.X11-unix/X${display#:}"
for _ in $(seq 1 100); do
    if [ -e "$socket" ]; then
        break
    fi
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
        echo "Xvfb exited before becoming ready; see $log" >&2
        exit 1
    fi
    sleep 0.1
done

DISPLAY="$display" "$@"
