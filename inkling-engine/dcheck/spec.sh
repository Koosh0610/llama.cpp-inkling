#!/bin/bash
# usage: spec.sh LABEL [server args...] -> runs llama-server (llama-ds2 build) with the args under the GPU lock, then spec_bench.py
LABEL=$1; shift
BIN=${BIN:-$HOME/src/llama-ds2/build/bin/llama-server}
M=$HOME/models/inkling2-mini/mainline-Inkling-2-Mini-UD-Q3_K_XL-f16ctrl.gguf
PORT=18089
run() {
  $BIN -m $M -fa on -c 4096 -ngl 99 --port $PORT "$@" > /private/tmp/claude-501/-Users-kushjuvekar/f2b09734-e90c-4999-87d2-b0619a1a7cdd/scratchpad/server_$LABEL.log 2>&1 &
  S=$!
  for i in $(seq 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 1; done
  python3 -I $HOME/src/inkling-engine/dcheck/spec_bench.py $PORT "$LABEL"
  kill $S; wait $S 2>/dev/null
}
export -f run 2>/dev/null
$HOME/src/inkling-engine/gpulock.sh bash -c "$(declare -f run); BIN=$BIN M=$M PORT=$PORT LABEL=$LABEL run ${*:+$(printf '%q ' "$@")}"
