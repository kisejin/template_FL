#!/bin/bash
# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')
# Create the logs directory with the datetime subdirectory
mkdir -p logs/$datetime

# Record start time
start_time=$(date +%s)
start_time_formatted=$(date '+%Y-%m-%d %H:%M:%S')

# Determine the path to the flwr package
flwr_path=$(python -c "import flwr; print(flwr.__file__)")
flwr_common_message_path=$(dirname $flwr_path)/common/message.py

# Change DEFAULT_TTL in the determined Python file
sed -i 's/DEFAULT_TTL = [0-9]\+/DEFAULT_TTL = 360000/' $flwr_common_message_path

# Run the command and save output and error logs
# Modify pyproject.toml to set mates.state to false
sed -i 's/^\(mates\.state\s*=\s*\).*/\1true/' pyproject.toml

CUDA_VISIBLE_DEVICES=2 flwr run . > logs/$datetime/output.log 2> logs/$datetime/error.log

# Record end time
end_time=$(date +%s)
end_time_formatted=$(date '+%Y-%m-%d %H:%M:%S')
duration=$((end_time - start_time))

# Convert seconds to hours, minutes, and seconds
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

# Log the timing information
# echo "Start time: $(date -d @${start_time} '+%Y-%m-%d %H:%M:%S')" >> logs/$datetime/timing.log
# echo "End time: $(date -d @${end_time} '+%Y-%m-%d %H:%M:%S')" >> logs/$datetime/timing.log
# echo "Total duration: ${hours}h ${minutes}m ${seconds}s" >> logs/$datetime/timing.log

echo "Start time: $start_time_formatted" >> logs/$datetime/timing.log
echo "End time: $end_time_formatted" >> logs/$datetime/timing.log
echo "Total duration: ${hours}h ${minutes}m ${seconds}s" >> logs/$datetime/timing.log