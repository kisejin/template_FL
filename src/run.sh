#!/bin/bash
# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')
# Create the logs directory with the datetime subdirectory
mkdir -p logs/$datetime

# Record start time
start_time=$(date +%s)

# Run the command and save output and error logs
CUDA_VISIBLE_DEVICES=3 flwr run . > logs/$datetime/output.log 2> logs/$datetime/error.log

# Record end time
end_time=$(date +%s)
duration=$((end_time - start_time))

# Convert seconds to hours, minutes, and seconds
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

# Log the timing information
echo "Start time: $(date -d @${start_time} '+%Y-%m-%d %H:%M:%S')" >> logs/$datetime/timing.log
echo "End time: $(date -d @${end_time} '+%Y-%m-%d %H:%M:%S')" >> logs/$datetime/timing.log
echo "Total duration: ${hours}h ${minutes}m ${seconds}s" >> logs/$datetime/timing.log