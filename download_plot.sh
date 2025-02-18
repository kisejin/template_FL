#!/bin/bash

# Download *.memmap from dropbox
wget -O folder_proj.zip "https://www.dropbox.com/scl/fi/7heiftp16cikpy1rm5e5u/folder_proj.zip?rlkey=xhprw5hmfyib8dxzozejk51jb&st=x92qrec2&dl=1"

# Extract file in zip
unzip folder_proj.zip

# Remove *.zip file
rm *.zip

# Move content in folder after zipped to folder src
find folder_proj/ -maxdepth 1 -type f -exec mv {} src/ \;

# Remove folder_proj
rm -rf folder_proj/