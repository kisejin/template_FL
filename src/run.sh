# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')

# Define variables for state and update
state=true
update=3

# Create the logs directory with the datetime subdirectory
mkdir -p logs/$datetime


# Record start time
start_time=$(date +%s)
start_time_formatted=$(date '+%Y-%m-%d %H:%M:%S')

# Determine the path to the flwr package
flwr_path=$(python -c "import flwr; print(flwr.__file__)")
flwr_common_message_path=$(dirname $flwr_path)/common/message.py

# Change DEFAULT_TTL in the determined Python file
sed -i 's/DEFAULT_TTL = [0-9]\+/DEFAULT_TTL = 100000/' $flwr_common_message_path

# Modify pyproject.toml to set mates.state and mates.num-data-influence-model-update
sed -i "s/^\(mates\.state\s*=\s*\).*/\1$state/" pyproject.toml

# Change folder parameters in utils.py to result_metric2
sed -i "s#\(folder=\"result_metric/\)[^\"]*\(\")\)#\1${datetime}\2#g" fedllm/utils.py

# Run the command and save output and error logs for the first run
CUDA_VISIBLE_DEVICES=0 flwr run . > logs/$datetime/output.log 2> logs/$datetime/error.log


# Record end time
end_time=$(date +%s)
end_time_formatted=$(date '+%Y-%m-%d %H:%M:%S')
duration=$((end_time - start_time))

# Convert seconds to hours, minutes, and seconds
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "Start time: $start_time_formatted" >> logs/$datetime/timing.log
echo "End time: $end_time_formatted" >> logs/$datetime/timing.log
echo "Total duration: ${hours}h ${minutes}m ${seconds}s" >> logs/$datetime/timing.log