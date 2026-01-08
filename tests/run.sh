#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Install test dependencies if needed (quietly)
pip install -q requests

# Run the flow test
echo "Running Integration Tests..."
python tests/test_flow.py
