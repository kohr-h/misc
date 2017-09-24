import numpy as np
import torch
from torch import autograd, nn
from tensorboardX import SummaryWriter
import scipy.misc


class Perceptron(nn.Module):

    def __init__(self, num_feat, bias=True):
        super(Perceptron, self).__init__()
        assert num_feat > 0

        self.linear = nn.Linear(num_feat, 1, bias)

    def forward(self, x):
        linear = self.linear(x)
        return nn.functional.sigmoid(linear)

    def __repr__(self):
        return '{}({}, bias={})'.format(self.__class__.__name__,
                                        self.linear.in_features,
                                        self.linear.bias is not None)


# %% Testing

# Generate training data
training_data = [
    (np.array([0, 0, 1]), 0),
    (np.array([0, 1, 1]), 1),
    (np.array([1, 0, 1]), 1),
    (np.array([1, 1, 1]), 1),
    ]


def xy_from_tdata(data):
    xs = [entry[0] for entry in data]
    ys = [entry[1] for entry in data]
    x = np.vstack(xs)
    y = np.vstack(ys)
    return torch.Tensor(x), torch.Tensor(y)


inputs_tens, targets_tens = xy_from_tdata(training_data)
inputs = autograd.Variable(inputs_tens)
targets = autograd.Variable(targets_tens)

# Set up network
model = Perceptron(3)
loss_func = nn.MSELoss(size_average=False)
params = list(model.parameters())

# For tensorboard, evaluate loss once to make the graph
writer = SummaryWriter()
loss = loss_func(model(inputs), targets)
writer.add_graph(model, loss)
# add random image just for testing
image = torch.Tensor(scipy.misc.ascent())
writer.add_image('ascent', image)

# Parameters for learning
learning_rate = 0.2
niter = 100


# Function mapping output to predicted labels
def predicted_labels(x):
    if isinstance(x, autograd.Variable):
        x = x.data

    out = torch.zeros_like(x)
    out[x >= 0.5] = 1
    return out


def num_misclassified(labels, true_labels):
    if isinstance(labels, autograd.Variable):
        labels = labels.data
    if isinstance(true_labels, autograd.Variable):
        true_labels = true_labels.data

    return int(torch.sum(labels != true_labels))


# Train the network
for it in range(1, niter + 1):
    # Compute loss
    out_pred = model(inputs)
    loss = loss_func(out_pred, targets)

    # Backpropagate using simple gradient descent
    model.zero_grad()
    loss.backward()
    for p in params:
        p.data.sub_(learning_rate * p.grad.data)

    if it % 10 == 0:
        print('Iteration {:>3}'.format(it))
        print('-------------')
        print('loss function value:', loss.data[0])
        pred_labels = predicted_labels(out_pred)
        print('predicted labels:', pred_labels)
        print('# misclassified:', num_misclassified(pred_labels, targets))
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), it)
        writer.add_scalar('loss', loss, it)


# %% Test with new data

# This is evil test data because the network had no chance to really learn
# what the third component does -- (nothing, at least in the training data)
test_data = [
    (np.array([0, 0, 0]), 0),
    (np.array([0, 1, 0]), 1),
    (np.array([1, 0, 0]), 1),
    (np.array([1, 1, 0]), 1),
    ]

test_inputs_tens, test_targets_tens = xy_from_tdata(test_data)
test_inputs = autograd.Variable(test_inputs_tens)
test_targets = autograd.Variable(test_targets_tens)

test_out_pred = model(test_inputs)
test_labels_pred = predicted_labels(test_out_pred)
print('predicted labels (test):', test_labels_pred)
print('# misclassified (test):',
      num_misclassified(test_labels_pred, test_targets))


# Don't forget to close the writer
writer.close()
