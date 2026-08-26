#!/bin/bash

echo Eternity-Code
echo Start training...

set -e

NPU_COUNT=$(npu-smi info -l 2>/dev/null | grep -E "^\|[ ]*[0-9]+ \|" | wc -l)
[ "$NPU_COUNT" -eq 0 ] && NPU_COUNT=1

torchrun --nproc_per_node=$NPU_COUNT ./main.py