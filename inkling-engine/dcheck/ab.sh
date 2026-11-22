#!/bin/bash
# usage: ab.sh "ENV_A=1" "ENV_B=1" [rounds] -> interleaved tg64 runs, mean and spread per side
A=$1; B=$2; N=${3:-8}
M=${M:-$HOME/models/inkling2-mini/mainline-Inkling-2-Mini-UD-Q3_K_XL-f16ctrl.gguf}
BIN=$HOME/src/llama.cpp/build/bin/llama-bench
for i in $(seq $N); do
  for side in A B; do
    e=$A; [ $side = B ] && e=$B
    v=$($HOME/src/inkling-engine/gpulock.sh env $e $BIN -m $M -fa 1 -p 0 -n 64 -r 1 2>&1 | grep tg64 | awk -F'|' '{print $(NF-1)}' | awk '{print $1}')
    echo "$side $v"
  done
done | awk '{s[$1]+=$2; q[$1]+=$2*$2; n[$1]++} END{for (k in s) {m=s[k]/n[k]; printf "%s mean %.2f sd %.2f n %d\n", k, m, sqrt(q[k]/n[k]-m*m), n[k]}}' | sort
