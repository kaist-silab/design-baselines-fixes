#!/bin/bash
 
echo "Adding lib to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

echo "Adding the libnvinfer library to the LD_LIBRARY_PATH"
PYTHON_DIR=$(dirname $(which python)); PARENT_DIR=$(dirname "$PYTHON_DIR")
LIBVINFER=$(find "$PARENT_DIR" -name "*libnvinfer.so.8")
LIBVINFER_PLUGIN=$(find "$PARENT_DIR" -name "*libnvinfer_plugin.so.8")
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LIBVINFER%/*} ## the one down does work, not sure why the one on the left does not
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/miniconda3/envs/design-baselines/lib/python3.7/site-packages/tensorrt

echo "Exporting Mujoco path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin

echo "Exporting the XLA flags from /usr/lib/cuda; you may need to change this path if you have a different CUDA location"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda   

echo "Checking if TensorFlow can detect GPU. If some error occurs, check the installation manually."
python -c 'import tensorflow as tf; tf.test.is_gpu_available(cuda_only=False)'
