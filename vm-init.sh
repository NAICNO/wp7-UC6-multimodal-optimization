#!/bin/bash
# Initial VM setup for NAIC Orchestrator VMs
# Run this ONCE on a fresh VM before cloning the repository

set -e

echo "=== NAIC VM Initial Setup ==="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Check for module system (NAIC VMs with EasyBuild/Lmod)
echo "Checking environment..."
if command -v module &> /dev/null; then
    echo "Module system detected (EasyBuild/Lmod)"
    echo "Available Python modules:"
    module avail python/3 2>&1 | head -20 || true
    echo ""
else
    echo "No module system detected - installing system packages..."
    $SUDO apt update -y
    $SUDO apt install -y \
        build-essential \
        git \
        gdb \
        libssl-dev \
        zlib1g-dev \
        python3-dev \
        python3-venv \
        python3-pip \
        htop \
        tmux
fi

# Check GPU availability
echo ""
echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || nvidia-smi
elif command -v rocm-smi &> /dev/null; then
    echo "AMD GPU detected:"
    rocm-smi
else
    # Check if GPU hardware exists but drivers not installed
    if lspci | grep -iE 'vga|3d|display' | grep -i nvidia &> /dev/null; then
        echo "NVIDIA GPU hardware detected but drivers not installed"
        lspci | grep -iE 'vga|3d|display'
    else
        echo "No GPU detected (CPU-only mode)"
    fi
fi

# Check Python
echo ""
echo "=== Python Status ==="
python3 --version 2>/dev/null || echo "Python3 not found"
which python3 2>/dev/null || echo "Python3 path not found"

echo ""
echo "=== VM Initial Setup Complete ==="
echo ""
echo "Next steps:"
echo "  git clone https://github.com/NAICNO/wp7-UC6-multimodal-optimization.git"
echo "  cd wp7-UC6-multimodal-optimization"
echo "  ./setup.sh"
echo "  source venv/bin/activate"
echo ""
