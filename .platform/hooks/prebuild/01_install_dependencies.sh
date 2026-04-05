#!/bin/bash
# Clean up temporary files
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*

# Set pip to use no cache and temporary directory with more space
export TMPDIR=/var/app/staging/tmp
mkdir -p $TMPDIR

# Install with optimized settings
/var/app/venv/staging-LQM1lest/bin/pip install --no-cache-dir --tmp $TMPDIR -r /var/app/staging/requirements.txt
