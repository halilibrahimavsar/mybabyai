#!/bin/bash
# scripts/run_tests.sh

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_ROOT"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: venv directory not found at $PROJECT_ROOT/venv"
    exit 1
fi

# Activate venv and run pytest
source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"
python -m pytest "$@"
