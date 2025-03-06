#!/bin/bash

PID=TARGET_PID_TO_MONITOR
COMMAND="bash examples/grpo_trainer/run_qwen2-7b_seq_balance.sh"

# Loop to check if the process with the given PID exists
while true; do
    if ! ps -p $PID > /dev/null; then
        echo "Process $PID has terminated. Launching training script..."
        $COMMAND
        break
    fi
    sleep 10  # Wait for 10 seconds before checking again
    echo "Waiting for process $PID to finish..."
done