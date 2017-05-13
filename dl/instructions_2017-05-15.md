# Deep Learning tasks

These are instructions for the deep learning experimentation session in our seminar on 2017-05-15.

**Important:** Please take notes while trying different things. Best would be to work in one *Python script* per task (not directly in the console) and adding comments as you go.
You can also work in a Jupyter notebook, just run (in a directory higher than where you want to put the notebook file):

    conda install notebook
    jupyter-notebook

### Defining custom Theano Operators

Although there's a lot of functionality already built into Theano (and even more into blocks), we certainly need to integrate our own code at some point, e.g., the ray transform.

[This guide](http://deeplearning.net/software/theano/extending/extending_theano.html) for defining a new Theano Operator that can later be used as a symbolic expression. We will not go through all details but instead start from a skeleton file that can be adapted to concrete needs. Furthermore, we go through the attributes of the `theano.Op` class that are most important for our purposes.

### Projects

As a more long-term effort, we want to use deep learning to answer research questions and develop new methods for all kinds of things in the field of inverse problems. Here are some ideas for projects that could get us started.

**Note:** These projects are likely not pure programming tasks but involve a bit of mathematics and conceptual thinking. Some are a bit open-ended and don't have a clear "best solution".

#### Implement NNFBP

- Need FBP as an Op, parametrized by the filter
- Need to define the Jacobian, aka the operator derivative wrt the filter
- Construct a simple CNN to append to a bunch of parallel FBP layers
- Generate training data: eiter data-phantom pairs or data-reco pairs where reco is using a method to mimic

#### Learn the regularization parameter

- Goal: map input data to regularization parameter, given a certain reconstruction method (e.g. TV)
- A step could be the definition of key figures from data (like noise level) to reduce input size
- Define figures of merit that comprise a cost function (L2 error, structural similarity etc.)
- Training data: simulated

#### Implement a scale-invariant convolution

- Find a way to define convolution kernels in terms of analytic functions that can be scaled with the problem
- A possiblity would be dilated and shifted Gaussians, another alternative wavelets
- Parameters would be the translations and dilations
- Compute kernel as multiplier in Fourier space, using dilation and translation properties of the FT
- Check how to do FFT within Theano (find API)

#### Segmentation

- Build a network for segmentation

#### Imaging with coupled channels

- Infer the (unknown) relation between the channels
