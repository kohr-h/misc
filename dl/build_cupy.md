# Building cupy from source

**Make sure to adapt the paths to your system.**

1. Make a custom `nvcc` script that sets `-ccbin` and uses `ccache` (optional):

       #!/bin/sh

       ccache /usr/local/cuda/bin/nvcc -ccbin /opt/sw/gcc-5.4.0/bin -Wno-deprecated-gpu-targets "$@"

2. Install dependencies:

       conda install six numpy cudnn nccl
       conda install -c conda-forge cython  # need version 0.27 or higher
       pip install fastrlock

3. Use the following command to compile `cupy`:

       CC=/opt/sw/gcc-5.4.0/bin/gcc \
       CXX=/opt/sw/gcc-5.4.0/bin/g++ \
       NVCC=~/local/bin/nvcc \
       CUDA_PATH=/usr/local/cuda \
       CFLAGS="-I$CONDA_PREFIX/include -I/opt/sw/gcc-5.4.0/include/" \
       LDFLAGS="-L$CONDA_PREFIX/lib" \
       LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
       pip install -e .
