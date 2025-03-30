#!/bin/bash

# Update package lists and install system dependencies
apt-get update && apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    libjack-dev \
    libpulse-dev

# Upgrade pip
pip install --upgrade pip
# Install project dependencies
# Install Python dependencies
pip install -r requirements-dev.txt
