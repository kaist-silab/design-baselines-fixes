#!/bin/bash

echo "Downloading mujoco200 from https://www.roboti.us/download/mujoco200_linux.zip"
wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco200_linux.zip
mkdir -p ~/.mujoco
unzip mujoco200_linux.zip -d ~/.mujoco
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
rm mujoco200_linux.zip

echo "Downloading mjkey.txt from https://www.roboti.us/file/mjkey.txt"
wget https://www.roboti.us/file/mjkey.txt -O ~/.mujoco/mjkey.txt


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin" >> ~/.bashrc

echo "Installing mujoco-py and its dependencies"
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
pip install Cython==0.29.21
pip install mujoco-py==2.0.2.3


echo "Creating the Conda environment"
conda env create -f environment.yml
conda activate design-baselines

echo "Installing CUDA and its dependencies"
pip install nvidia-pyindex nvidia-tensorrt
 
echo "Linking libnvinfer.so.8 to libnvinfer.so.7 and its plugin"
PYTHON_DIR=$(dirname $(which python)); PARENT_DIR=$(dirname "$PYTHON_DIR")
LIBVINFER=$(find "$PARENT_DIR" -name "*libnvinfer.so.8")
LIBVINFER_PLUGIN=$(find "$PARENT_DIR" -name "*libnvinfer_plugin.so.8")
ln -s $LIBVINFER ${LIBVINFER%/*}/libnvinfer.so.7
ln -s $LIBVINFER_PLUGIN ${LIBVINFER_PLUGIN%/*}/libnvinfer_plugin.so.7

echo "Adding the libnvinfer library to the LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LIBVINFER%/*}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LIBVINFER_PLUGIN%/*}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${LIBVINFER%/*}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib" >> ~/.bashrc

echo "Exporting the XLA flags from /usr/lib/cuda; you may need to change this path if you have a different CUDA location"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda   
echo "export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda" >> ~/.bashrc

echo "Checking if TensorFlow can detect GPU. If some error occurs, check the installation manually."
python -c 'import tensorflow as tf; tf.test.is_gpu_available(cuda_only=False)'