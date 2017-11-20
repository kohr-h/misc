# Parallelism in pyTorch and Chainer

We investigate the possibility to parallelize training of deep neural networks using pyTorch and Chainer. Generally, we distinguish between **data parallelism** and **model parallelism**, and in the following we document the support of the frameworks for this feature and which approach(es) are implemented in the library cores.

We also check for an opportunity to "serialize" parallel GPU code, in the sense that we use a single GPU but process data in chunks that are sent to the GPU one at a time. This "fake parallelism" is useful for datasets that are too large to be processed in parallel (batches / channels) on a single GPU. It would allow to use existing models with minimal changes, while significantly reducing the GPU memory required to evaluate and train the model.


## Principle

For **data parallelism**, we seek to **split the input data** into chunks that can be sent to different GPUs for processing. For **model parallelism**, we evaluate different parts of the computation graphs on different nodes (GPUs, machines).

Let us assume our input data is a tensor `x` of shape `(B, 1, N)`, where the first axis of size `B` is the batch axis, the second one of size 1 the channel axis, and the remaining axes of shape `N` the "proper" data axes. If we evaluate, for instance, a batched convolution with `C` output channels, the overall logic looks roughly like this:

    x1[1, 1, N] --+
                  |
        ...     --+--> x[B, 1, N_in] ---{Conv[K_1, ..., K_C]}---> y[B, C, N_out]
                  |
    xB[1, 1, N] --+

Here, `K_i` is the convolution kernel for the i-th channel. Suppose we have `G` GPUs. In this scenario we can distinguish the following strategies:

### 1. Splitting along the batch axis
The input `x` is split along the first axis, and the inputs are sent to different GPUs and processed independently, like this:

    (CPU)   x[B, 1, N] --+                                                               +--> y[B, C, N_out]
                         |                                                               |
    (GPU 1)              +--> x1[b, 1, N] ---{Conv[K_1, ..., K_C]}---> y1[b, C, N_out] --+
                         |                                                               |
                         +-->  ...                                                ...  --+
                         |                                                               |
    (GPU G)              +--> xG[b, 1, N] ---{Conv[K_1, ..., K_C]}---> y1[b, C, N_out] --+

Here `b` refers to the chunk size, assuming `B = b * G` exactly for simplicity. Since all GPUs perform the same work, this is a typical case of pure **data parallelism**.

**Advantage:** Every batch is processed the same way (as per definition of "batch"), so each GPU can use the original network model. In practice, this is achieved by replicating the model on each node and running them independently. In the backpropagation, gradients are computed independently as well, and at the last step gathered into a single update.

**Disadvantage:** Often the size `(b, 1, N)` of each chunk, or the output size `(b, C, N_out)`, will still be too large to be processed at once. Further chunking is needed in that case.

**Partial mitigation:** On each GPU, batches can be split up further, e.g., such that the batch size becomes 1. This lowers the required amount of memory per evaluation to `N` for the input and `(C, N_out)` for the output. This will, however, often still be too large since it is not uncommon to have, e.g., `C = 32` channels.


### 2. Splitting along the channel axis
Assuming that we process "un-batched" data, i.e., data with shape `(1, ...)`, we further split the data along the channel axis. In the case considered here, this affects the output only, but in general, the input can also already have multiple channels.
The main difference is that each GPU no longer runs the same code:

    (CPU)   x[1, 1, N] --+                                                                             +--> y[1, C, N_out]
                         |                                                                             |
    (GPU 1)              +--> x1[1, 1, N] ---{Conv[K_(0*c+1),     ..., K_(1*c)]}---> y1[1, c, N_out] --+
                         |                                                                             |
                         +-->  ...                                                              ...  --+
                         |                                                                             |
    (GPU G)              +--> xG[1, 1, N] ---{Conv[K_((G-1)*c+1), ..., K_(G*c)]}---> y1[1, c, N_out] --+

In this diagram, `c` refers to the chunked channel size, assuming that `C = c * G`. Hence, GPU `i` performs a convolution with the kernels `K_((i-1)*c+1), ..., K_(i*c)`. Hence, this splitting type belongs to the category of **model parallelism**.

**Advantage:** This splitting mode allows us to further reduce the size of the data that each GPU has to handle at a time. If the sizes `N` and `(c, N_out)` are still too large, we can further split each channel chunk to single-channel convolutions using the same technique (but not broadcasting to GPUs).

**Disadvantage:** We need to create a separate model (or module/layer) for each GPU. This requires code that understands which layers have channels, and how to replace them with reduced layers for parts of the channels only. For built-in modules supporting channels, this is doable (should not be much more than convolution and matrix multiplication), but custom layers with channels will not be supported.


### 3. Splitting in the data axes
Now we assume that we can deal with one dataset of size `(1, 1, N)` and can evaluate on one channel, producing output of size `(1, 1, N_out)`. If we want to further chunk up the data, we need to compute convolutions on blocks and merge the results:

    (CPU)   x1[1, 1, n] --+                                                   +--> y1[1, 1, n_out] --+--> y[1, 1, N_out]
                          |                                                   |                      |
    (GPU)                 +--> x1[1, 1, n] ---{Conv[K]}---> y1[1, 1, n_out] --+                      |
                                                                                                     |
    ...                                         ...                                          ...   --+
                                                                                                     |
    (CPU)   xM[1, 1, n] --+                                                   +--> yM[1, 1, n_out] --+
                          |                                                   |
    (GPU)                 +--> xM[1, 1, n] ---{Conv[K]}---> yM[1, 1, n_out] --+

Here we assume that `N = n * M`, where `M` is the number of blocks per axis.

**Advantage:** For this model we can evaluate blocks of size `n` at a time, and by selecting `M` appropriately we can make sure that this is a computable size.

**Disadvantage:** This operation is not equivalent to the original convolution of the whole tensor. The implicit zero-padding at the boundary of each block will result in wrong results close to the boundary, where "close" means "less than `kernel_size / 2` away". Even for small kernels, e.g., shape `(3, ..., 3)`, the intermediate layers will have to deal with wrong values (on a lattice-shaped region) that break the translation invariance an may influence the final results.

**Mitigation:** The regions should overlap enough to always use correct values for the convolution and not use any implicit zero-padding. This requires some logic to split up and merge the blocks correctly, but has the advantage that the convolution can be used as-is.


## Support in pyTorch

`torch` has wrapper classes for existing models that implement parallelism. TODO: write more

- [`DataParallel` class](http://pytorch.org/docs/master/nn.html#dataparallel) single-machine
- [`DistributedDataParallel` class](http://pytorch.org/docs/master/nn.html#distributeddataparallel) distributed

## Support in Chainer

`chainer` seems to have a more low-level approach to parallelism that provides more opportunity to customize parallelization. TODO: extend this

- [Tutorial on model parallelism with multiple GPUs](https://docs.chainer.org/en/stable/tutorial/gpu.html#model-parallel-computation-on-multiple-gpus)
- [Tutorial on data parallelism with mutliple GPUs and `Trainer`](https://docs.chainer.org/en/stable/tutorial/gpu.html#data-parallel-computation-on-multiple-gpus-with-trainer)
- [Tutorial on data parallelism with mutliple GPUs, without `Trainer`](https://docs.chainer.org/en/stable/tutorial/gpu.html#data-parallel-computation-on-multiple-gpus-without-trainer)
