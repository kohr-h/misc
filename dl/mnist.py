from theano import tensor
from blocks.bricks import Linear, Rectifier, Softmax, MLP
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.cost import MisclassificationRate

# Input variable
x = tensor.matrix('features')

# Define layers: input -> linear -> ReLU -> linear -> softmax -> output
# The `layer.apply()` method makes the connection to previous output
input_to_hidden = Linear(name='input_to_hidden', input_dim=784, output_dim=100)
h = Rectifier().apply(input_to_hidden.apply(x))
hidden_to_output = Linear(name='hidden_to_output',
                          input_dim=100, output_dim=10)
y_hat = Softmax().apply(hidden_to_output.apply(h))

# Output variable
y = tensor.lmatrix('targets')

# Build MLP directly (replaces y_hat). Don't forget initialization!
mlp_model = MLP(activations=[Rectifier(), Softmax()],
                dims=[784, 100, 10],
                weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
mlp_model.initialize()
mlp = mlp_model.apply(x)

# Define cost function (switch y_hat for mlp)
cost = CategoricalCrossEntropy().apply(y.flatten(), mlp)
# Build the computation graph. This is probably needed to get out the
# weights for the later definition of the final cost function.
# Also add the misclassification rate for later monitoring. It needs to
# be added here so it actually gets computed.
misclass = MisclassificationRate().apply(y.flatten(), mlp)
cg = ComputationGraph([cost, misclass])
# `VariableFilter` is a class that filters `cg.variables` (a list)
W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
# The final cost, an L2-regularized cross entropy loss. As usual, the
# regularization parameter is a magical constant.
cost = (cost +
        0.005 * (W1 ** 2).sum() +
        0.005 * (W2 ** 2).sum())
cost.name = 'cost_with_regularization'

# Initialize the weights and biases
input_to_hidden.weights_init = IsotropicGaussian(0.01)
hidden_to_output.weights_init = IsotropicGaussian(0.01)
input_to_hidden.biases_init = Constant(0)
hidden_to_output.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_output.initialize()

# Import the dataset. This requires the data to be available on disk in
# the data folder.
mnist = MNIST(("train",))

# Define the data stream from the data file to the algorithm. We flatten
# the images (so the input becomes a linear vector), iterate sequentially
# (as opposed to what?) and use batches of size 256 (probably this is
# the batch size as used in SGD).
data_stream = Flatten(DataStream.default_stream(
    mnist,
    iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))

# Use gradient descent for minimization
algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                            step_rule=Scale(learning_rate=0.1))

# Import the test data and make it available as as stream, too.
mnist_test = MNIST(("test",))
data_stream_test = Flatten(DataStream.default_stream(
    mnist_test,
    iteration_scheme=SequentialScheme(
        mnist_test.num_examples, batch_size=1024)))

# Define an object that serves to monitor the cost function applied to the
# test data.
monitor = DataStreamMonitoring(
    variables=[cost, misclass], data_stream=data_stream_test, prefix="test")

# The main loop connects all loose ends. The data stream is connected
# to the algorithm and (what happens to the extensions?)
main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                     extensions=[monitor, FinishAfter(after_n_epochs=4),
                                 Printing()])
main_loop.run()
