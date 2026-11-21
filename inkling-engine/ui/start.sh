#!/bin/bash
# starts the UI on :8090; it launches upstream (:18101) and ours (:18102) on demand, one at a time
cd "$(dirname "$0")"
mkdir -p logs
pgrep -f "inkling-engine/ui/server.py" > /dev/null || nohup python3 -I $PWD/server.py > logs/ui.log 2>&1 &
sleep 1; echo "http://127.0.0.1:8090"
