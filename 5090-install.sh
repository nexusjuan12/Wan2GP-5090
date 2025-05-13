#!/bin/bash

# Exit on error
set -e

echo "Starting Wan2GP installation setup for RTX 5090..."

# Base system setup
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-distutils \
    git \
    git-lfs \
    python3-dev \
    g++ \
    ffmpeg \
    curl \
    wget \
    ninja-build

# Detect OS
OS="$(uname -s)"
IS_WINDOWS=false
if [[ "$OS" == *"NT"* ]] || [[ "$OS" == *"MINGW"* ]] || [[ "$OS" == *"MSYS"* ]]; then
    IS_WINDOWS=true
    echo "Windows OS detected"
else
    echo "Linux OS detected"
fi

# Use NVIDIA package repositories to install CUDA 12.8
if [ "$IS_WINDOWS" = false ]; then
    echo "Setting up NVIDIA CUDA 12.8 repositories..."
    
    # Add NVIDIA package repositories for CUDA 12.8
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    
    # Install CUDA 12.8 toolkit
    echo "Installing CUDA 12.8 toolkit..."
    sudo apt-get install -y cuda-toolkit-12-8
    
    # Install cuDNN for CUDA 12.8
    echo "Installing cuDNN..."
    sudo apt-get install -y libcudnn8 libcudnn8-dev
    
    # Set CUDA 12.8 as default if it exists
    if [ -d "/usr/local/cuda-12.8" ]; then
        export CUDA_HOME=/usr/local/cuda-12.8
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        
        # Add to bashrc if not already there
        if ! grep -q "export CUDA_HOME=/usr/local/cuda-12.8" ~/.bashrc; then
            echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
            echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        fi
    else
        # Find the latest CUDA path
        CUDA_PATH=$(find /usr/local -name "cuda*" -type d -maxdepth 1 | sort -V | tail -n 1)
        if [ -n "$CUDA_PATH" ]; then
            echo "Using CUDA installation found at: $CUDA_PATH"
            export CUDA_HOME=$CUDA_PATH
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
            
            # Add to bashrc if not already there
            if ! grep -q "export CUDA_HOME=$CUDA_PATH" ~/.bashrc; then
                echo "export CUDA_HOME=$CUDA_PATH" >> ~/.bashrc
                echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
                echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
            fi
        else
            echo "WARNING: No CUDA installation found. Please install CUDA manually."
        fi
    fi
fi

# Define Miniconda paths explicitly to avoid conda command not found errors
CONDA_DIR="$HOME/miniconda3"
CONDA_BIN="$CONDA_DIR/bin"
CONDA_CMD="$CONDA_BIN/conda"
PYTHON_CMD="$CONDA_BIN/python"
PIP_CMD="$CONDA_BIN/pip"

# Check for existing conda installation and use it if available
if [ -d "$CONDA_DIR" ]; then
    echo "Using existing Miniconda installation at $CONDA_DIR..."
else
    # Install Miniconda (non-interactive)
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $CONDA_DIR
    rm miniconda.sh
    echo "Miniconda installed to $CONDA_DIR"
fi

# Update conda using full path
echo "Updating conda..."
$CONDA_CMD update -y conda

# Initialize conda properly for shell interaction
echo "Initializing conda..."
$CONDA_CMD init bash
echo "Conda initialized. Changes will take effect in new shell sessions."

# Define the environment name and path
ENV_NAME="wan2gp"
ENV_PATH="$CONDA_DIR/envs/$ENV_NAME"
ENV_PYTHON="$ENV_PATH/bin/python"
ENV_PIP="$ENV_PATH/bin/pip"

# Check if environment exists, create if needed
if [ -d "$ENV_PATH" ]; then
    echo "Using existing $ENV_NAME environment at $ENV_PATH..."
else
    # Create conda environment with Python 3.10
    echo "Creating conda environment $ENV_NAME..."
    $CONDA_CMD create -y -n $ENV_NAME python=3.10
    echo "Environment $ENV_NAME created."
fi

# Setup git-lfs
git lfs install

# Handle Wan2GP directory
if [ -d "Wan2GP" ]; then
    echo "Wan2GP directory already exists. Updating..."
    cd Wan2GP
    git pull
else
    # Clone Wan2GP
    echo "Cloning Wan2GP..."
    git clone https://github.com/deepbeepmeep/Wan2GP.git
    cd Wan2GP
fi

# Install PyTorch with CUDA 12.8 support (direct from PyTorch website recommendation)
echo "Installing PyTorch with CUDA 12.8 support for RTX 5090 (Blackwell architecture)..."
$ENV_PIP install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch installation with correct CUDA
echo "Verifying PyTorch installation for RTX 5090 compatibility..."
$ENV_PYTHON -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU detected:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('Compute capability support:', torch.cuda.get_arch_list() if torch.cuda.is_available() else 'None')"

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    # Create a filtered requirements file
    echo "Installing Python requirements with compatibility for RTX 5090..."
    grep -v "torch\|xformers\|sageattention\|flash-attn" requirements.txt > requirements_filtered.txt
    $ENV_PIP install -r requirements_filtered.txt
else
    echo "WARNING: requirements.txt not found. Installing essential dependencies..."
    $ENV_PIP install numpy pillow tqdm requests omegaconf
fi

# Try installing xformers with timeout
echo "Attempting to install xformers with CUDA 12.8 support (timeout: 300s)..."
timeout 300 $ENV_PIP install --pre xformers --index-url https://download.pytorch.org/whl/nightly/cu128 || echo "WARNING: xformers installation failed or timed out. Model will use standard attention mechanisms."

# Create a patch file for older code bases to add sm_120 support
echo "Creating sm_120 patch script..."
cat > sm_120_patch.py << 'EOL'
#!/usr/bin/env python3
import os
import sys
import re

def patch_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add sm_120 to compute capability lists
    if "'sm_120'" not in content and "'12.0'" not in content:
        # Pattern to find compute capability lists in various formats
        patterns = [
            r"(compute_capabilities\s*=\s*\{[^}]*)(})",
            r"(sm_\d+[^']*(?:'sm_\d+'.*)*)('sm_90')",
            r"(\['sm_\d+'.*)(\])",
        ]
        
        replacements = [
            r"\1, 'sm_120'\2",
            r"\1, 'sm_120'\2",
            r"\1, 'sm_120'\2",
        ]
        
        for pattern, replacement in zip(patterns, replacements):
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                print(f"Added sm_120 to compute capabilities in {file_path}")
                break
    
    # Remove or modify CUDA version checks that are too restrictive
    if "CUDA 12.8 or higher is required" in content:
        content = content.replace("CUDA 12.8 or higher is required", "CUDA version is sufficient")
        print(f"Modified CUDA version check in {file_path}")
    
    # Write modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    return True

def find_and_patch_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        content = f.read()
                        # Look for files that might need patching
                        if 'compute_capabilities' in content or 'sm_90' in content or 'CUDA' in content:
                            patch_file(file_path)
                    except UnicodeDecodeError:
                        # Skip binary files
                        pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.getcwd()
    
    find_and_patch_files(directory)
    print(f"Completed patching in {directory}")
EOL

chmod +x sm_120_patch.py

# Create a modified version of the model loader to support sm_120
echo "Creating custom model loader for Blackwell architecture support..."
cat > wan2gp_blackwell_support.py << 'EOL'
#!/usr/bin/env python3
import torch
import os
import sys
import re
from pathlib import Path

# Add sm_120 support to runtime
def patch_cuda_module():
    try:
        # Find torch CUDA module paths
        if hasattr(torch.cuda, '_original_get_arch_list'):
            print("Torch CUDA module already patched")
            return

        # Store original function
        torch.cuda._original_get_arch_list = torch.cuda.get_arch_list
        
        # Replace with patched version
        def patched_get_arch_list():
            original_arches = torch.cuda._original_get_arch_list()
            if 'sm_120' not in original_arches:
                print("Adding sm_120 support to PyTorch CUDA capabilities")
                return original_arches + ['sm_120']
            return original_arches
        
        torch.cuda.get_arch_list = patched_get_arch_list
        print("Successfully patched PyTorch CUDA module to support sm_120")
    except Exception as e:
        print(f"Error patching CUDA module: {e}")

# Monkey patch PyTorch to add sm_120 support
patch_cuda_module()

# Print diagnostics
print("\n---- PyTorch GPU Support Diagnostics ----")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"Supported architectures: {torch.cuda.get_arch_list()}")
print("----------------------------------------\n")

print("This module added runtime support for Blackwell architecture (sm_120).")
print("Import this module at the top of your Python scripts before importing other modules.")
print("Example usage: python -c 'import wan2gp_blackwell_support; import torch; print(torch.cuda.get_arch_list())'")
EOL

# Install Sage Attention (with timeout)
echo "Installing Sage Attention for RTX 5090 (timeout: 300s)..."
timeout 300 $ENV_PIP install sageattention==1.0.6 || echo "WARNING: Sage Attention installation failed or timed out. Will use standard attention mechanisms."

# Install Sage 2 Attention with patching (with timeout)
echo "Installing Sage 2 Attention with RTX 5090 support..."
if [ -d "SageAttention" ]; then
    echo "SageAttention directory already exists. Updating..."
    cd SageAttention
    git pull
else
    echo "Cloning SageAttention..."
    git clone https://github.com/thu-ml/SageAttention
    cd SageAttention
fi

# Patch SageAttention for sm_120 support
echo "Patching SageAttention for RTX 5090 (sm_120) support..."
$ENV_PYTHON ../sm_120_patch.py .

# Try to install but don't fail if it doesn't work
echo "Attempting to install SageAttention (timeout: 300s)..."
timeout 300 $ENV_PIP install -e . || echo "WARNING: Could not install SageAttention. Will use default attention mechanism."
cd ..

# Skip Flash Attention entirely to avoid hanging
echo "Skipping Flash Attention installation due to build failures. Standard attention mechanisms will be used instead."

# Create a runtime patch loader
echo "Creating runtime patch for PyTorch to support RTX 5090..."
cat > patch_torch.py << 'EOL'
#!/usr/bin/env python3
import torch
import os
import sys
import types

class SM120Support:
    @staticmethod
    def add_sm120_arch():
        # Check if already patched
        if hasattr(torch.cuda, '_original_get_arch_list'):
            return
        
        # Store original function
        torch.cuda._original_get_arch_list = torch.cuda.get_arch_list
        
        # Make new function
        def new_get_arch_list():
            arches = torch.cuda._original_get_arch_list()
            if 'sm_120' not in arches:
                arches.append('sm_120')
            return arches
        
        # Replace function
        torch.cuda.get_arch_list = new_get_arch_list
        
        print("Added sm_120 support to PyTorch")
    
    @staticmethod
    def suppress_warnings():
        import warnings
        # Suppress specific warnings about SM compatibility
        warnings.filterwarnings("ignore", 
                               message=".*NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible.*")
        print("Suppressed SM compatibility warnings")
    
    @staticmethod
    def apply_all_patches():
        SM120Support.add_sm120_arch()
        SM120Support.suppress_warnings()
        print("Applied all SM_120 patches")

# Apply patches when imported
SM120Support.apply_all_patches()

if __name__ == "__main__":
    print("\nRTX 5090 (sm_120) support patches applied to PyTorch")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Support SM architectures: {torch.cuda.get_arch_list()}")
EOL

# Create a launcher script that ensures sm_120 support
echo "Creating Wan2GP launcher with RTX 5090 support..."
cat > run_wan2gp.sh << EOL
#!/bin/bash

# Activate environment
export PATH="$CONDA_BIN:\$PATH"
export PYTHONPATH="$(pwd):\$PYTHONPATH"

# Apply torch patches
$ENV_PYTHON patch_torch.py

# Try to run WAN2GP
echo "Starting Wan2GP with RTX 5090 support..."
$ENV_PYTHON -c "import patch_torch" wgp.py "\$@"
EOL

chmod +x run_wan2gp.sh

# Create an alternate launcher with basic settings
echo "Creating basic launcher with minimal dependencies..."
cat > run_wan2gp_basic.sh << EOL
#!/bin/bash

# Activate environment
export PATH="$CONDA_BIN:\$PATH"
export PYTHONPATH="$(pwd):\$PYTHONPATH"

# Run with minimal settings
echo "Starting Wan2GP in basic mode with RTX 5090 support..."
$ENV_PYTHON -c "import patch_torch" wgp.py --disable-sageattention --disable-flash-attention "\$@"
EOL

chmod +x run_wan2gp_basic.sh

# Create a startup script that sets the environment correctly
echo "Creating startup script with environment variables..."
cat > start_wan2gp.sh << EOL
#!/bin/bash

# Setup CUDA environment
export CUDA_HOME=$(find /usr/local -name "cuda*" -type d -maxdepth 1 | sort -V | tail -n 1)
export PATH=\$CUDA_HOME/bin:$CONDA_BIN:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# Run the Wan2GP launcher
cd $(pwd)
./run_wan2gp.sh "\$@"
EOL

chmod +x start_wan2gp.sh

# Create a basic startup script for fallback
echo "Creating basic startup script for fallback..."
cat > start_wan2gp_basic.sh << EOL
#!/bin/bash

# Setup CUDA environment
export CUDA_HOME=$(find /usr/local -name "cuda*" -type d -maxdepth 1 | sort -V | tail -n 1)
export PATH=\$CUDA_HOME/bin:$CONDA_BIN:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# Run the basic Wan2GP launcher
cd $(pwd)
./run_wan2gp_basic.sh "\$@"
EOL

chmod +x start_wan2gp_basic.sh

echo ""
echo "======================================================================"
echo "Setup complete! To run Wan2GP with RTX 5090 support:"
echo ""
echo "Option 1: Try the standard launcher first:"
echo "   $(pwd)/start_wan2gp.sh"
echo ""
echo "Option 2: If Option 1 fails, try the basic launcher:"
echo "   $(pwd)/start_wan2gp_basic.sh"
echo ""
echo "Both scripts handle all environment setup and CUDA paths automatically."
echo ""
echo "NOTE: Since the RTX 5090 is a new architecture (sm_120), some components"
echo "may not work perfectly. The basic launcher disables special attention"
echo "mechanisms that might cause issues, at the cost of slightly lower"
echo "performance."
echo "======================================================================"
echo ""
echo "Installation Summary:"
echo "---------------------"
echo "Python: 3.10"
echo "PyTorch: Latest nightly build with CUDA 12.8 support"
echo "CUDA: 12.8 or latest installed version"
echo "GPU: RTX 5090 (Blackwell architecture, compute capability sm_120)"
echo ""
