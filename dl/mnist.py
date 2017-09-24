from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import autograd, nn
import torchvision
from tensorboardX import SummaryWriter

# %% Show example images

data_root = os.path.join(os.environ['HOME'], 'git', 'misc',
                         'dl', 'data', 'mnist')
dset_train = torchvision.datasets.MNIST(
    root=data_root, train=True, download=True,
    )

# Show 4 random images with labels from the dataset
indcs = np.random.randint(0, len(dset_train) - 1, 4)
examples = [dset_train[i] for i in indcs]
fig, ax = plt.subplots(ncols=4)
for ax_i, ex_i in zip(ax, examples):
    ax_i.imshow(ex_i[0], cmap='gray')
    ax_i.set_title('label = {}'.format(ex_i[1]))
fig.suptitle('Example MNIST images')
fig.tight_layout()
plt.show()

# %% Load dataset as tensors

dset_train = torchvision.datasets.MNIST(
    root=data_root, train=True, download=True,
    transform=torchvision.transforms.ToTensor()
    )


# %% Define neural network

# Local response normalization, see
# https://github.com/pytorch/pytorch/issues/653#issuecomment-326851808
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75,
                 across_channels=False):
        super(LRN, self).__init__()
        self.across_channels = across_channels

        if self.across_channels:
            self.average = nn.AvgPool3d(
                kernel_size=(local_size, 1, 1),
                stride=1,
                padding=((local_size - 1) // 2, 0, 0))
        else:
            self.average = nn.AvgPool2d(
                kernel_size=local_size,
                stride=1,
                padding=(local_size - 1) // 2)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.across_channels:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Flatten(nn.Module):

    def __init__(self, dims=None):
        super(Flatten, self).__init__()

        if dims is None:
            dims = (1, 2, 3)

        try:
            iter(dims)
        except TypeError:
            self.dims = (int(dims),)
        else:
            self.dims = tuple(d for d in sorted(dims))

    def forward(self, x):
        view_shape = [x.shape[i] for i in range(len(x.shape))
                      if i not in self.dims]
        return x.view(view_shape + [-1])


model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    LRN(),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    LRN(),
    Flatten(),
    nn.Linear(1600, 128),
    nn.Tanh(),
    nn.Dropout(0.8),
    nn.Linear(128, 256),
    nn.Tanh(),
    nn.Dropout(0.8),
    nn.Linear(256, 10),
    nn.Softmax(),
    )


# %% Optimizer and loss function

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fun = nn.CrossEntropyLoss()

# %% Train the network

num_epochs = 10
batch_size = 1000
num_batches = len(dset_train) // batch_size
assert num_batches * batch_size == len(dset_train)

model.cuda()

writer = SummaryWriter(comment='_mnist')

for epoch in range(1, num_epochs + 1):
    print('*************')
    print('* EPOCH {:>3} *'.format(epoch))
    print('*************')

    running_loss = 0
    model.zero_grad()

    for batch in range(num_batches):
        data = dset_train.train_data[batch * batch_size:
                                     (batch + 1) * batch_size]
        data = data.type(torch.FloatTensor)[:, None, ...]
        inputs = autograd.Variable(data.cuda())  # remove .cuda() for CPU
        labels = dset_train.train_labels[batch * batch_size:
                                         (batch + 1) * batch_size]
        targets = autograd.Variable(labels.cuda())

        loss = loss_fun(model(inputs), targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.data[0]
        writer.add_scalar('loss', running_loss / (batch + 1))
        print('[{:>3}, {:>3}]: loss = {}'.format(epoch,
                                                 (batch + 1) * batch_size,
                                                 running_loss / (batch + 1)))


# %% Validate

dset_test = torchvision.datasets.MNIST(
    root=data_root, train=False, download=True,
    transform=torchvision.transforms.ToTensor()
    )
indcs = np.random.randint(0, len(dset_test) - 1, 4)
validation_set = [dset_test[i] for i in indcs]
predicted_energies = [model(autograd.Variable(inp[0][None, :]).cuda())
                      for inp in validation_set]
predicted_labels = [torch.max(x, 1)[1].data[0] for x in predicted_energies]
true_labels = [data[1] for data in validation_set]
for lbl, true in zip(predicted_labels, true_labels):
    print('predicted: {}, true {}'.format(lbl, true))
