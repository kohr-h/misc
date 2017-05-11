import numpy
import theano.tensor as tt
from theano import function, shared
rng = numpy.random

# Training sample size
N = 400
# number of input variables
feats = 784

# Generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# --- Declare Theano symbolic variables --- #

X = tt.dmatrix('X')
y = tt.dvector('y')

# Initialize the weight vector w randomly
#
# This and the following bias variable b are shared so they keep their values
# between training iterations (updates)
w = shared(rng.randn(feats), name='w')

# Initialize the bias term
b = shared(0., name='b')

print('Initial model:')
print('w =', w.get_value())
print('b =', b.get_value())

# --- Construct Theano expression graph --- #

# Probability that target = 1
p_1 = 1 / (1 + tt.exp(-tt.dot(X, w) - b))
# The prediction thresholded
prediction = p_1 > 0.5
# Cross-entropy loss function
xent = -y * tt.log(p_1) - (1 - y) * tt.log(1 - p_1)
# The cost to minimize - l2 regularized cross entropy
cost = xent.mean() + 0.01 * (w ** 2).sum()
# Compute the gradient of the cost w.r.t weight vector w and bias term b
# (we shall return to this in a following section of this tutorial)
gw, gb = tt.grad(cost, [w, b])

# Compile
train = function(
    inputs=[X, y],
    outputs=[prediction, xent],
    updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

predict = function(inputs=[X], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print('w =', w.get_value())
print('b =', b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
pred = predict(D[0]).astype(int)
print(pred)
print('Prediction errors: at indices')
print(numpy.where(pred != D[1])[0])
