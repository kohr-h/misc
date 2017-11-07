# Neural Networks in 3D - memory limit tests

Report on memory limits of deep learning frameworks for a very simple network for 3D input.

## Setup

The test setup consists of a 3-layer network with 3D convolutions having channels `1 -> 4`, `4 -> 8` and `8 -> 1`, respectively. After each convolution, a `ReLU` unit is applied. The final step is the computation of the mean squared error against a target of appropriate size. A graph of the test network is available at [this link](https://github.com/kohr-h/misc/blob/master/dl/test_model_3d.pdf).
Furthermore, a backward pass is evaluated to compute gradients.

The input is of size `(N, N, N)`, where `N` is varied to probe the memory limits. A naive count of channels in the network gives an expected memory usage of about 4 + 8 + 1 = 13 times the input for the convolutions and the same amount for the nonlinearities, assuming that memory is persistent for all intermediate results.

## Results for `chainer`

The results here are obtained with source installations of [`chainer`](https://github.com/chainer/chainer) (commit 29c4536) and [`cupy`](https://github.com/cupy/cupy) (commit bad4182), with all dependencies (cuDNN, NCCL, ...). The code corresponding to this report is [here](https://github.com/kohr-h/misc/blob/master/dl/chainer_test.py).

The following table shows the memory usage of the network itself, the input and target variables, and the intermediate nodes in the forward and backward passes, respectively, as read off `nvidia-smi` after the corresponding steps. Numbers in parentheses are `mem_used / (26 * input_size)`, which should be close to 1 according to the naive reasoning above. Memory sizes are in MB.

N   | Network | Variables | Forward     | Backward
----|---------|-----------|-------------|---------
50  | 2       | 2         | 118 (9.1)   | 8
64  | 2       | 2         | 134 (5.1)   | 8
100 | 2       | 6         | 196 (1.9)   | 8
128 | 2       | 14        | 290 (1.3)   | 60
200 | 2       | 60        | 808 (0.97)  | 230
256 | 2       | 124       | 1592 (0.91) | ERROR (1)
300 | 2       | 202       | 2500 (0.89) | ERROR (1)
400 | 2       | 480       | 5824 (0.88) | ERROR (1)
450 | 2       | 682       | 6886 (0.73) | OOM
500 | 2       | 938       | OOM         | OOM

(1): The call to `cupy.cuda.cudnn.convolutionBackwardData_v3` fails with `CuDNNError: CUDNN_STATUS_NOT_SUPPORTED: b'CUDNN_STATUS_NOT_SUPPORTED'`. Probably the convolution algorithm for this size hasn't been implemented yet.

**Conclusions:**
- The initialization of the CUDA backend uses about 100 MB of GPU memory.
- For `N` between 64 and 100, there must a convolution algorithm switch since the relative memory usage of the forward pass drops sharply. Another switch is between 200 and 256 since the backwards algorithm (also a convolution) starts failing.
- For small `N`, the memory overhead in the forward pass is enormous (several times the "naive" size).
- Even for larger `N`, the memory usage is not significantly less than `total_num_outputs * input_size`. It probably approaches `num_conv_outputs * input_size` since the `ReLU` outputs can be dropped, while the convolution outputs need to be retained for the backward pass.
- The backward pass does not add significant memory overhead. Probably only the intermediate `ReLU` gradients are added, although they could also be dropped immediately after being used.


## Results for `torch`

Results are obtained with a source installation of [`pytorch`](https://github.com/pytorch/pytorch) (commit 561fc8d), with all dependencies (cuDNN, NCCL, ...). The code corresponding to this report is [here](https://github.com/kohr-h/misc/blob/master/dl/torch_test.py).

The table is analogous to the one for `chainer`.

N   | Network | Variables | Forward     | Backward
----|---------|-----------|-------------|---------
50  | <1 (2)  | <1 (2)    | 134 (10.3)  | 4
64  | <1 (2)  | <1 (2)    | 150 (5.5)   | 10
100 | <1 (2)  | 8         | 218 (2.1)   | 52
128 | <1 (2)  | 16        | 324 (1.5)   | 110
200 | <1 (2)  | 60        | 1090 (1.3)  | 230
256 | <1 (2)  | 124       | 2180 (1.3)  | ERROR (3)
300 | <1 (2)  | 202       | 3448 (1.2)  | ERROR (3)
350 | <1 (2)  | 320       | 5438 (1.2)  | ERROR (3)
400 | <1 (2)  | 480       | OOM         | OOM

(2): Probably the CUDA init allocates a chunk of memory and uses that for small amounts of data. Therefore the network and the variables take no extra memory for small `N`.

(3) Similar error as in `cupy` in the cuDNN backend. Seems like backward 3D convolution is not implemented for `N = 256` and larger.

**Conclusions:**
- The initialization of the CUDA backend uses about 280 MB of GPU memory.
- Behavior is largely the same as for `chainer`, with worse constants. The memory usage never reaches less than the "naive" size. The out-of-memory point is reached already for `N = 350`.
