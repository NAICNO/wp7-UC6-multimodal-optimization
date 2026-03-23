#!/bin/bash
# Setup script for NAIC Multi-Modal Optimization (SHGA)
# Run this after cloning: ./setup.sh

set -e

echo "=== NAIC Multi-Modal Optimization Setup ==="
echo ""

# Initialize git submodules (CEC2013 benchmarks)
if [ -d ".git" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
    echo "Submodules initialized"
    echo ""
fi

# Fix CEC2013 upstream bug (get_info() missing self. prefix)
if [ -f "fix-cec2013.sh" ]; then
    echo "Applying CEC2013 patch..."
    ./fix-cec2013.sh
    echo ""
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "ERROR: Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "Python version OK"

# Check for GPU
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "GPU detected but nvidia-smi query failed"
else
    echo "No NVIDIA GPU detected (CPU-only mode)"
fi

# Create virtual environment
echo ""
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    echo "Reusing existing venv. To recreate: rm -rf venv && ./setup.sh"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Create activation wrapper script with PYTHONPATH
echo ""
echo "Creating activation wrapper script..."
cat > activate-mmo.sh << 'WRAPPER_EOF'
#!/bin/bash
# Activate MMO environment with PYTHONPATH configured
# Usage: source activate-mmo.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/venv/bin/activate"
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}:${SCRIPT_DIR}/benchmarks/CEC2013/python3"
echo "MMO environment activated (PYTHONPATH configured)"
WRAPPER_EOF
chmod +x activate-mmo.sh
echo "Created activate-mmo.sh"

# Set for current session
PROJECT_DIR=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}:${PROJECT_DIR}/benchmarks/CEC2013/python3"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source activate-mmo.sh"
echo ""
echo "To run the demonstrator notebook:"
echo "  jupyter lab"
echo ""
echo "Quick test:"
echo "  python -c \"from mmo.minimize import MultiModalMinimizer; print('MMO: OK')\""
echo "  python -c \"from cec2013.cec2013 import CEC2013; print('CEC2013: OK')\""
echo ""
