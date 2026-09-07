#!/bin/bash
# Stop execution if any command fails
set -e

ENV_NAME="sbi_script_env"

echo "========================================"
echo "1. Initializing Conda and Creating Env"
echo "========================================"
# Ensure conda activate works inside the script
eval "$(conda shell.bash hook)"

# Create the environment with Python 3.10 and necessary dependencies
conda create -y -n $ENV_NAME -c conda-forge \
    python=3.12 \
    gsl \
    fftw \
    c-compiler \
    cxx-compiler \
    fortran-compiler \
    make \
    pkg-config \
    git \
    numpy \
    matplotlib \
    scipy \
    jupyter \
    pyyaml \
    emcee \
    tqdm \
    pytorch \
    sbi

# Activate the environment
conda activate $ENV_NAME

echo "========================================"
echo "2. Configuring Environment Variables"
echo "========================================"
# Set variables for the current session to ensure compilation works
export CPATH="$CONDA_PREFIX/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Make these variables persistent whenever the environment is activated
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat <<EOF > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
export OLD_CPATH="\$CPATH"
export OLD_LIBRARY_PATH="\$LIBRARY_PATH"
export OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"

export CPATH="\$CONDA_PREFIX/include:\$CPATH"
export LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LIBRARY_PATH"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
EOF

cat <<EOF > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
export CPATH="\$OLD_CPATH"
export LIBRARY_PATH="\$OLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$OLD_LD_LIBRARY_PATH"

unset OLD_CPATH
unset OLD_LIBRARY_PATH
unset OLD_LD_LIBRARY_PATH
EOF

echo "========================================"
echo "3. Compiling MUSIC"
echo "========================================"
if [ ! -d "music" ]; then
    git clone https://bitbucket.org/ohahn/music.git music
else
    echo "MUSIC directory already exists. Pulling latest..."
    git -C music pull
fi

cd music
# Disable HDF5 in the Makefile as per the notebook settings
sed -i -e '/HAVEHDF5  /s/yes/no/' Makefile
make
cd ..

echo "========================================"
echo "4. Installing SCRIPT"
echo "========================================"
if [ ! -d "script" ]; then
    git clone https://bitbucket.org/rctirthankar/script script
else
    echo "SCRIPT directory already exists. Pulling latest..."
    git -C script pull
fi

cd script
pip install .
cd ..

echo "========================================"
echo "Setup Complete!"
echo "To use this environment, run: conda activate $ENV_NAME"
echo "========================================"
