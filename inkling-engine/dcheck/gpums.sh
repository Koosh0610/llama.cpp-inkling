#!/bin/bash
# usage: gpums.sh "ENV_A=1" "ENV_B=1" [rounds] -> interleaved, median GPU ms per decode token (needs TMPTIME hook)
A=$1; B=$2; N=${3:-4}
M=${M:-$HOME/models/inkling2-mini/mainline-Inkling-2-Mini-UD-Q3_K_XL-f16ctrl.gguf}
BIN=$HOME/src/llama.cpp/build/bin/llama-bench
for i in $(seq $N); do
  for side in A B; do
    e=$A; [ $side = B ] && e=$B
    env $e TMPTIME=1 $BIN -m $M -fa 1 -p 0 -n 64 -r 1 2>&1 | grep "TMPTIME nodes" | awk -v s=$side '
      / cb 1 /{st=$(NF-3)} / cb 0 /{print s, $(NF-1)-st}'
  done
done | sort -k1,1 -k2n | awk '{v[$1][++n[$1]]=$2} END{for (k in n) {m=v[k][int(n[k]/2)+1]; printf "%s median %.3f ms/token (%.2f tok/s GPU-bound), n %d\n", k, m, 1000/m, n[k]}}'
