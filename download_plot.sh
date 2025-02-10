#!/bin/bash

# Download *.memmap from dropbox
wget -O folder_proj.zip "https://www.dropbox.com/scl/fi/jqkx0elgvn86jvmpwsblo/250101_Hop-dong-dich-vu.docx?rlkey=l7481gvf4tr7zfls2q2p4ztr1&st=rdh2r2ph&dl=1"

# Extract file in zip
unzip folder_proj.zip

# Remove *.zip file
rm *.zip

# Move content in folder after zipped to folder src
find ~/folder_proj -maxdepth 1 -type f -exec mv {} ~/src \;