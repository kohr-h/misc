import matplotlib.pyplot as plt
import numpy as np
import odl
import odl.contrib.torch as odl_torch
import torch
from torch import nn, autograd
from tensorboardX import SummaryWriter

# %% Random ellipse generator

max_ab = 0.5
space = space = odl.uniform_discr([-1, -1], [1, 1], (128, 128),
                                  dtype='float32')
pts = space.points()


def ellipse(fval, center, a, b, phi):
    test_pts = pts - np.array(center)[None, :]
    rot_mat = np.array([[np.cos(phi), -np.sin(phi)],
                        [np.sin(phi), np.cos(phi)]])
    test_pts = rot_mat.T.dot(test_pts.T).T
    inside = np.where(
        (test_pts[:, 0] / a) ** 2 + (test_pts[:, 1] / b) ** 2 <= 1)[0]
    ell = np.zeros(space.shape)
    ell.ravel()[inside] = fval
    return ell.astype('float32')


def random_ellipses(n):
    assert n >= 1

    fval = np.random.uniform(0.1, 2)
    center = np.random.uniform(-1, 1, size=2)
    a, b = np.random.uniform(0.05, max_ab, size=2)
    phi = np.random.uniform(0, np.pi)

    ell = ellipse(fval, center, a, b, phi)

    for _ in range(n - 1):
        fval = np.random.uniform(0.1, 2)
        center = np.random.uniform(-1, 1, size=2)
        a, b = np.random.uniform(0.05, max_ab, size=2)
        phi = np.random.uniform(0, np.pi)

        ell += ellipse(fval, center, a, b, phi)

    return ell.astype('float32')


# %% Define ray transform

angles = odl.uniform_partition(0, np.pi, 50)
det_part = odl.uniform_partition(-2, 2, 256)
geometry = odl.tomo.Parallel2dGeometry(angles, det_part)
ray_trafo = odl.tomo.RayTransform(space, geometry)

# %% Define network, loss function and optimizer


class NNFBP(nn.Module):

    def __init__(self, fbp_ker_size=11):
        super(NNFBP, self).__init__()

        self.fbp_ker_size = fbp_ker_size
        # Data space convolution only along axis 1
        self.conv_det = nn.Conv2d(1, 1, kernel_size=(1, self.fbp_ker_size),
                                  padding=(0, self.fbp_ker_size - 1),
                                  bias=False)

        # Backprojection layer
        self.backproj = odl_torch.OperatorAsModule(ray_trafo.adjoint)

        # Volumetric convolutions, one full and one reducing to 1 channel
        # self.conv1_vol = nn.Conv2d(32, 32, 3, padding=2)
        # self.conv2_vol = nn.Conv2d(32, 1, 3, padding=2)

    def forward(self, x):
        y = self.conv_det(x)[..., :-(self.fbp_ker_size - 1)]
        y = self.backproj(y)
        # y = self.conv1_vol(y)
        # y = y[..., :-2, :-2]
        # y = nn.functional.sigmoid(y)
        # y = self.conv2_vol(y)
        # y = y[..., :-2, :-2]
        # y = nn.functional.sigmoid(y)
        return y


loss_func = nn.MSELoss(size_average=False)

# %% Train the network

num_epochs = 1
train_size = 2
learn_rate = 0.00000000000001
cuda = True
use_tb = False

nnfbp = NNFBP()
if cuda:
    nnfbp = nnfbp.cuda()

if use_tb:
    # Compute loss once to make a graph
    writer = SummaryWriter(comment='_nnfbp')
    target_arr = random_ellipses(np.random.randint(5, 40))
    target = autograd.Variable(torch.from_numpy(target_arr))
    if cuda:
        target = target.cuda()

    inp_arr = ray_trafo(target_arr)
    inp = autograd.Variable(torch.from_numpy(inp_arr.asarray()))
    if cuda:
        inp = inp.cuda()
    inp = inp[None, None, ...]
    outp = nnfbp(inp)
    loss = loss_func(outp, target)
    writer.add_graph(nnfbp, loss)


for epoch in range(1, num_epochs + 1):
    print('*************')
    print('* EPOCH {:>3} *'.format(epoch))
    print('*************')

    with odl.util.NumpyRandomSeed(123):
        for i in range(train_size):
            target_arr = random_ellipses(np.random.randint(5, 40))
            fig, ax = plt.subplots()
            ax.imshow(target_arr, cmap='gray')
            ax.set_title('Epoch {}, image {}, ground truth'.format(epoch, i))
            fig.show()
            target = autograd.Variable(torch.from_numpy(target_arr))
            if cuda:
                target = target.cuda()

            inp_arr = ray_trafo(target_arr)
            inp = autograd.Variable(torch.from_numpy(inp_arr.asarray()))
            if cuda:
                inp = inp.cuda()
            inp = inp[None, None, ...]

            outp = nnfbp(inp)

            inp_npy = inp.data.cpu().numpy()
            out_npy = outp.data.cpu().numpy()

            fig, ax = plt.subplots()
            ax.imshow(inp_npy.squeeze(), cmap='gray')
            ax.set_title('Epoch {}, image {}, input'.format(epoch, i))
            fig.show()

            fig, ax = plt.subplots()
            ax.imshow(out_npy.squeeze(), cmap='gray')
            ax.set_title('Epoch {}, image {}, output'.format(epoch, i))
            fig.show()


# Optimizer
optimizer = torch.optim.SGD(nnfbp.parameters(), lr=learn_rate)

for epoch in range(1, num_epochs + 1):
    print('*************')
    print('* EPOCH {:>3} *'.format(epoch))
    print('*************')

    running_loss = 0

    # Use seed to make sure that the same data is created in each epoch
    with odl.util.NumpyRandomSeed(123):
        for i in range(train_size):
            target_arr = random_ellipses(np.random.randint(5, 40))
            target = autograd.Variable(torch.from_numpy(target_arr))
            if cuda:
                target = target.cuda()

            inp_arr = ray_trafo(target_arr)
            inp = autograd.Variable(torch.from_numpy(inp_arr.asarray()))
            if cuda:
                inp = inp.cuda()
            inp = inp[None, None, ...]

            outp = nnfbp(inp)
            nnfbp.zero_grad()
            loss = loss_func(outp, target)
            loss.backward()

            optimizer.step()
            running_loss += loss.data[0]
            if i % 10 == 0:
                print('[{:>3}, {:>4}] loss = {:.4}'.format(
                         epoch, i, running_loss))
                if use_tb:
                    writer.add_scalar('loss', running_loss,
                                      (epoch - 1) * train_size + i)

writer.close()


# %%  Display parameters

def show_kernel(kernel):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
