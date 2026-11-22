#!/bin/bash
# usage: gpulock.sh <command...> -> runs the command holding the shared GPU lock (one timing run at a time)
L=$HOME/src/inkling-engine/.gpu.lock
until mkdir "$L" 2>/dev/null; do sleep 5; done
trap 'rmdir "$L"' EXIT
"$@"
