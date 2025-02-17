#!/bin/bash

# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')

# Create the logs directory with the datetime subdirectory
mkdir -p logs/$datetime

# Determine the path to the flwr package
flwr_path=$(python -c "import flwr; print(flwr.__file__)")
flwr_common_message_path=$(dirname $flwr_path)/common/message.py

# Change DEFAULT_TTL in the determined Python file
sed -i 's/DEFAULT_TTL = [0-9]\+/DEFAULT_TTL = 100000/' $flwr_common_message_path
sed -i 's/^\(mates\.num-data-influence-model-update\s*=\s*\).*/\1 5/' pyproject.toml
# Change folder parameters in utils.py to result_metric2
sed -i "s|folder=\"result_metric\"|folder=\"result_metric/$datetime\"|g" fedllm/utils.py
# Run the command and save output and error logs for the first run
flwr run . > logs/$datetime/output.log 2> logs/$datetime/error.log
wait


# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')

# Create the logs directory with the datetime subdirectory
mkdir -p logs/$datetime
sed -i 's/^\(mates\.num-data-influence-model-update\s*=\s*\).*/\1 3/' pyproject.toml
# Change folder parameters in utils.py to result_metric2
sed -i "s|folder=\"result_metric\"|folder=\"result_metric/$datetime\"|g" fedllm/utils.py
# Run the command again after modification
flwr run . > logs/$datetime/output.log 2> logs/$datetime/error.log
wait


# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')

# Create the logs directory with the datetime subdirectory
mkdir -p logs/$datetime
sed -i 's/^\(mates\.num-data-influence-model-update\s*=\s*\).*/\1 10/' pyproject.toml
# Change folder parameters in utils.py to result_metric3
sed -i "s|folder=\"result_metric\"|folder=\"result_metric/$datetime\"|g" fedllm/utils.py
# Run the command again after modification
flwr run . > logs/$datetime/output.log 2> logs/$datetime/error.log
wait


# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')

# Create the logs directory with the datetime subdirectory
mkdir -p logs/$datetime
# Modify pyproject.toml to set mates.state to false
sed -i 's/^\(mates\.state\s*=\s*\).*/\1false/' pyproject.toml
# Change folder parameters in utils.py to result_metric2
sed -i "s|folder=\"result_metric\"|folder=\"result_metric/$datetime\"|g" fedllm/utils.py
# Run the command again after modification
flwr run . > logs/$datetime/output.log 2> logs/$datetime/error.log
wait