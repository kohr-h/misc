# Deep Learning seminar 2017-05-09

These are instructions for the deep learning experimentation session in our seminar on 2017-05-09.

## Setting up a working environment

This guide requires `conda` to be installed. The first step is to create a fresh working environment with all the required packages:

    conda create -n theano python=3.6 nomkl numpy scipy six libgpuarray pyyaml toolz \
        picklable-itertools progressbar2 h5py pytables pyzmq pillow requests

    source activate theano

If you want to use MKL instead of `openblas` for linear algebra, remove the `nomkl` package.

The second step is to clone the repositories of a bunch of projects that we will need:

    git clone https://github.com/Theano/Theano
    git clone https://github.com/mila-udem/fuel.git
    git clone https://github.com/mila-udem/blocks.git
    git clone https://github.com/mila-udem/blocks-examples

The [`fuel` project](https://fuel.readthedocs.io/en/latest/) provides simple access to publicly available datasets on the internet. Its purpose is to download data, convert to a common format and deliver data as a stream to algorithms.
[`blocks`](https://blocks.readthedocs.io/en/latest/index.html) is a high-level learning framework built on top of Theano. It offers a variety of predefined building blocks, so-called "bricks" that make model building much easier than directly in Theano.
The `blocks-examples` repository simply contains a bunch of example scripts that map more or less to the examples in the Blocks tutorial.

Now we install these packages with the commands

    pip install Theano/
    pip install fuel/
    pip install blocks/

For convenient access to documentation, `spyder` is a good choice of IDE:

    conda install spyder

Finally, we need to tell `fuel` where to put the data. This can be done using the following command:

    echo 'data_path: "/path/to/data/"' > ~/.fuelrc

Now, for instance, we can download the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset:

    cd /path/to/data
    fuel-download mnist
    fuel-convert mnist

## Exercises and Experiments

**Important:** Please take notes while trying different things. Best would be to work in one *Python script* per task (not directly in the console) and adding comments as you go.
You can also work in a Jupyter notebook, just run (in a directory higher than where you want to put the notebook file):

    conda install notebook
    jupyter-notebook

### Theano

There's probably a heck of a lot to say about Theano, but we'll keep it to the basic things necessary to understand the higher-level structures later. We cover more or less the contents of [this tutorial](http://deeplearning.net/software/theano/tutorial/adding.html) and the following.

After this step, we should understand

- Theano variables (types, shapes, ...),
- Theano functions,
- the difference between symbolic and numerical computations,
- what happens when a function is created.

To visualize computation graphs, Theano has the built-in tool `pydotprint` that needs the `pydot-ng` package:

    conda install pyparsing
    pip install pydot-ng

Now we can export graphs like this:

```python
from theano.printing import pydotprint
pydotprint(theano_func, outfile='graph.png')
```

An alternative output format using the same backend is the `d3viz` module. It produces a simpler but also less complete representation of the graph in an HTML page that can be loaded in the browser:

```python
from theano.d3viz import d3viz
d3viz(theano_func, outfile='graph.html')
```

##### Tasks:

1. Follow the first page of the tutorial, keeping in mind which quantities are symbolic and which are numeric. Investigate the types of the variables by inspecting `obj.type`.
2. Build a function `g` that computes `sum(x**2 + 2*x*y + y**2)` for two vectors `x` and `y`, and another function `h` that computes `sum((x + y)**2)`. Evaluate the functions with numeric values to see if it works. Plot the computation graphs of both variants. What do you observe?
3. Check out the [guide on gradients](http://deeplearning.net/software/theano/tutorial/gradients.html) and understand how derivatives are defined. Compute the derivatives of our two functions `g` and `h` and display their computation graphs. (There's some more interesting stuff in this section, but we only need the basics for now.)


### Blocks

We now go one level higher and use layers, input and output directly as concepts instead of building everything by hand. That comes at the cost of some flexibility, but makes the learning curve much less steep. (Just compare [this MLP tutorial](http://deeplearning.net/tutorial/mlp.html#mlp) against [this one](https://blocks.readthedocs.io/en/latest/tutorial.html).)
We'll start with a simple and relatively small dataset.

#### MNIST dataset

[This dataset](http://yann.lecun.com/exdb/mnist/) consists in 60000 (training) + 10000 (test) hand-written single-digit numbers, each represented as 28x28 image. It is already quite old and small enough to do training and testing rather quickly.

##### Tasks:

1. Understand the structure of the program in the tutorial piece by piece. Map it to the mathematical description. Annotate the code (can also be questions).
   Typical things to play around with:
   - What are the types of the variables in the program? What are their values?
   - Which input do the functions and class constructors expect? What are additional options?
2. The text says one can directly define an MLP without manually defining and connecting connecting all the layers. Find out how to use the `mlp` object in the example code. Which variable does it replace?

   **Hint:** read the documentation of the `MLP` class.

   Convince yourself that `mlp` really does the same as the manually defined model (this may require a bit of digging).

3. Add another layer to the model. What parts of the code need to change?
4. Change the cost function to use L1 regularization in the weight coefficients.

#### Taking it further

Next we want to look into [Building with Bricks](https://blocks.readthedocs.io/en/latest/create_your_own_brick.html), which more or less explains the general concept of bricks, and [Create your own Brick](https://blocks.readthedocs.io/en/latest/create_your_own_brick.html), which introduces the API of the `Brick` class.

Think about relevant use cases, for instance how to rebuild the Neural Network FBP method with these concepts, or how to define a method for misalignment correction in tomography.
