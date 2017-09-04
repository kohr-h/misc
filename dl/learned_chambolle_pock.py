"""Learned chambolle-pock (PDHG) method."""

# from adler.tensorflow import prelu, cosine_decay
import numpy as np
import os
import odl
from odl.contrib.pytorch import TorchOperator
import torch


# --- Phantom generators --- #


def random_ellipse(interior=False):
    if interior:
        x_0 = np.random.rand() - 0.5
        y_0 = np.random.rand() - 0.5
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0

    return ((np.random.rand() - 0.5) * np.random.exponential(0.4),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc, n_ellipse=50, interior=False):
    n = np.random.poisson(n_ellipse)
    ellipses = [random_ellipse(interior=interior) for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, ellipses)


# --- Setup code --- #


np.random.seed(0)
try:
    name = os.path.splitext(os.path.basename(__file__))[0]
except NameError:
    # Interactive session
    name = os.path.join(os.getcwd(), 'tmp')
assert torch.cuda.is_available()

# --- Create ODL data structures --- #

size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator


# --- Create torch layer from ODL operator --- #


op_layer = TorchOperator(operator)
adj_layer = TorchOperator(operator.adjoint)

# --- Generate data --- #

n_data = 5


def generate_data(validation=False):
    """Generate a set of random data."""
    n_generate = 1 if validation else n_data

    # TODO: check what is needed here
    y_shape = (n_generate, operator.range.shape[0], operator.range.shape[1])
    y_arr = np.empty(y_shape, dtype='float32')
    x_shape = (n_generate, space.shape[0], space.shape[1])
    x_true_arr = np.empty(x_shape, dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = random_phantom(space)
        data = operator(phantom)
        noise = (odl.phantom.white_noise(operator.range) *
                 np.mean(np.abs(data)) * 0.05)
        noisy_data = data + noise

        x_true_arr[i, ...] = phantom
        y_arr[i, ...] = noisy_data

    return y_arr, x_true_arr


# --- Custom layers --- #


class PReLU(torch.nn.Module):

    def __init__(self, shape):
        super(PReLU, self).__init__()
        self.alphas = torch.nn.Parameter(torch.zeros(shape))
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        pos = self.relu(input)
        neg = -self.alphas * self.relu(-input)
        return pos + neg


class PdhgMethod(torch.nn.Module):

    """Module for the PDHG method with a fixed number of iterations.

    The PDHG method to solve the problem

    .. math::
        x^\dagger = \mathrm{arg\, min}_{x} f(Kx) + g(x)

    with convex :math:`f` and :math:`g` goes as follows:

    .. math::
        & \\text{Given:} & x_0, \\bar x_0, y_0, \\tau, \sigma, \\theta, N,
        n = 0 \\\\
        & \\text{While } & n \\leq N: \\\\
        && y_{n+1} \\leftarrow
            \mathrm{prox}_{\sigma f^*}(y_n + \sigma K \\bar x_n) \\\\
        && x_{n+1} \\leftarrow
            \mathrm{prox}_{\\tau g}(x_n - \\tau K^*y_{n+1}) \\\\
        && \\bar x_{n+1} \\leftarrow
            x_{n+1} + \\theta (x_{n+1} - x_n) \\\\
        && n \\leftarrow
            n + 1
    """

    def __init__(self, wrapped_op, wrapped_adj, num_iter):
        super(PdhgMethod, self).__init__()

        # Set non-learnable parameters
        self.wrapped_op = wrapped_op
        self.wrapped_adj = wrapped_adj
        self.num_iter = num_iter

        # Set learnable parameters with initial values
        self.sigma = torch.nn.Parameter(torch.FloatTensor(0.5))
        self.tau = torch.nn.Parameter(torch.FloatTensor(0.5))
        self.theta = torch.nn.Parameter(torch.FloatTensor(1.0))

        # Set layers, which are as follows:
        #
        # 1. apply operator
        # 2.
        pass


def apply_conv(x, filters=32):
    torch.nn.Conv2d()
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())


# Compute forward pass
primal = torch.autograd.Variable(torch.zeros(size, size))
primal_bar = torch.autograd.Variable(torch.zeros_like(primal.data))
dual = torch.autograd.Variable(torch.zeros(*operator.range.shape))



with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.zeros_like(x_true)
        primal_bar = tf.zeros_like(x_true)
        dual = tf.zeros_like(y_rt)

    for i in range(n_iter):
        with tf.variable_scope('dual_iterate',
                               reuse=True if i != 0 else None):
            evalop = odl_op_layer(primal_bar)
            # What is this??
            update = tf.concat([dual + sigma * evalop, y_rt], axis=-1)

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=1)
            dual = dual + update

        with tf.variable_scope('primal_iterate',
                               reuse=True if i != 0 else None):
            evalop = odl_op_layer_adjoint(dual)
            update = primal - tau * evalop

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=1)
            primal = primal + update

            primal_bar = primal + theta * update

    x_result = primal


with tf.name_scope('loss'):
    residual = x_result - x_true
    squared_error = residual ** 2
    loss = tf.reduce_mean(squared_error)


with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    maximum_steps = 100001
    starter_learning_rate = 1e-3
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta2=0.99)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        optimizer = opt_func.apply_gradients(zip(grads, tvars),
                                             global_step=global_step)


# Summaries
# tensorboard --logdir=...

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('psnr', -10 * tf.log(loss) / tf.log(10.0))

    tf.summary.image('x_result', x_result)
    tf.summary.image('x_true', x_true)
    tf.summary.image('squared_error', squared_error)
    tf.summary.image('residual', residual)

    merged_summary = tf.summary.merge_all()
    test_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/test',
                                                sess.graph)
    train_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/train')

# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

# Generate validation data
y_arr_validate, x_true_arr_validate = generate_data(validation=True)

if 0:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

# Train the network
for i in range(0, maximum_steps):
    if i%10 == 0:
        y_arr, x_true_arr = generate_data()

    _, merged_summary_result_train, global_step_result = sess.run([optimizer, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr,
                                         y_rt: y_arr,
                                         is_training: True})

    if i>0 and i%10 == 0:
        loss_result, merged_summary_result, global_step_result = sess.run([loss, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr_validate,
                                         y_rt: y_arr_validate,
                                         is_training: False})

        train_summary_writer.add_summary(merged_summary_result_train, global_step_result)
        test_summary_writer.add_summary(merged_summary_result, global_step_result)

        print('iter={}, loss={}'.format(global_step_result, loss_result))

    if i>0 and i%1000 == 0:
        saver.save(sess,
                   adler.tensorflow.util.default_checkpoint_path(name))
