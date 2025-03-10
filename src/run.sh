# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')

# Define variables for state and update
state=true
update=3

# Create the logs directory with the datetime subdirectory
mkdir -p logs/$datetime

# Determine the path to the flwr package
flwr_path=$(python -c "import flwr; print(flwr.__file__)")
flwr_common_message_path=$(dirname $flwr_path)/common/message.py

# Change DEFAULT_TTL in the determined Python file
sed -i 's/DEFAULT_TTL = [0-9]\+/DEFAULT_TTL = 100000/' $flwr_common_message_path

# Modify pyproject.toml to set mates.state and mates.num-data-influence-model-update
sed -i "s/^\(mates\.state\s*=\s*\).*/\1$state/" pyproject.toml
sed -i "s/^\(mates\.num-data-influence-model-update\s*=\s*\).*/\1$update/" pyproject.toml

# Change folder parameters in utils.py to result_metric2
sed -i "s#\(folder=\"result_metric/\)[^\"]*\(\")\)#\1${datetime}\2#g" fedllm/utils.py

# Run the command and save output and error logs for the first run
CUDA_VISIBLE_DEVICES=5 flwr run . > logs/$datetime/output.log 2> logs/$datetime/error.log