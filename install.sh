#!/bin/bash

# Choice for installing Mujoco: ask user
case "$1" in
    -y|--yes)
        echo "Installing Mujoco automatically"
        INSTALL_MUJOCO="y"
        ;;
    -n|--no)
        echo "Skipping Mujoco installation"
        INSTALL_MUJOCO="n"
        ;;
    *)
        echo "Do you want to install Mujoco automatically? (y/n)"
        read INSTALL_MUJOCO
        ;;
esac

if [ "$INSTALL_MUJOCO" = "y" ]; then
    echo "Downloading mujoco200 from https://www.roboti.us/download/mujoco200_linux.zip"
    wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco200_linux.zip
    mkdir -p ~/.mujoco
    unzip mujoco200_linux.zip -d ~/.mujoco
    mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
    rm mujoco200_linux.zip

    echo "Downloading mjkey.txt from https://www.roboti.us/file/mjkey.txt"
    wget https://www.roboti.us/file/mjkey.txt -O ~/.mujoco/mjkey.txt

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin

    echo "Installing mujoco-py and its dependencies"
    sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
    pip install Cython==0.29.21 mujoco-py==2.0.2.3 robel==0.1.2 morphing-agents==1.5.1 design-bench[all]==2.0.20
fi


echo "Creating the Conda environment"
conda env create -f environment.yml
conda init "$(basename $SHELL)"

echo "Reinitializing the shell and activating the Conda environment"
exec $SHELL
conda activate design-baselines

echo "Installing CUDA and its dependencies"
pip install nvidia-pyindex nvidia-tensorrt
 
echo "Linking libnvinfer.so.8 to libnvinfer.so.7 and its plugin"
PYTHON_DIR=$(dirname $(which python)); PARENT_DIR=$(dirname "$PYTHON_DIR")
LIBVINFER=$(find "$PARENT_DIR" -name "*libnvinfer.so.8")
LIBVINFER_PLUGIN=$(find "$PARENT_DIR" -name "*libnvinfer_plugin.so.8")
ln -s $LIBVINFER ${LIBVINFER%/*}/libnvinfer.so.7
ln -s $LIBVINFER_PLUGIN ${LIBVINFER_PLUGIN%/*}/libnvinfer_plugin.so.7


echo "Applying fix for importing design-bench; this works also with no installed Mujoco"
cp fixes/__init__.py $(python -c "import site; print(site.getsitepackages()[0] + '/design_bench/oracles/exact/')")

echo "Setting env vars in conda with LD_LIBRARY_PATH and XL_FLAGS; you may need to change this path if you have a different CUDA location"
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$(dirname $(dirname $(which python)))/lib:$(dirname $(find "$(dirname $(dirname $(which python)))" -name "*libnvinfer.so.8")):~/.mujoco/mujoco200/bin
conda env config vars set XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda

echo "Reactivating the Conda environment"
conda activate design-baselines
conda env config vars list

echo "Checking if TensorFlow can detect GPU. If some error occurs, check the installation manually."
python -c 'import tensorflow as tf; tf.test.is_gpu_available(cuda_only=False)'