#!/bin/bash

# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')

# Create the logs directory with the datetime subdirectory
mkdir -p logs/$datetime

# Run the command and save output and error logs
flwr run . > logs/$datetime/output.log 2> logs/$datetime/error.log