#!/bin/bash
# usage: check.sh -> compares 3 prompts against references
cd "$(dirname "$0")"; . ./prompts.sh
M=${M:-$HOME/models/inkling2-mini/mainline-Inkling-2-Mini-UD-Q3_K_XL-f16ctrl.gguf}; R=$HOME/models/inkling2-mini/eval/dref
for i in 1 2 3; do eval P=\$P$i; ../gpulock.sh ./dcheck $M "$P" 128 /private/tmp/claude-501/-Users-kushjuvekar/f2b09734-e90c-4999-87d2-b0619a1a7cdd/scratchpad/dcheck_out.bin $R/p$i.bin 2>&1 | tail -1 | sed "s/^/p$i: /"; done
