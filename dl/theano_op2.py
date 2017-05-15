import numpy as np
import odl
import theano


# This is the obvious mode: just wrap the ODL operator.
class TheanoOp(theano.Op):

    def __init__(self, odl_op):
        self.odl_op = odl_op

    # Properties attribute
    __props__ = ()

    # itypes and otypes attributes are
    # compulsory if make_node method is not defined.
    # They're the type of input and output respectively
    itypes = None
    otypes = None

    # Compulsory if itypes and otypes are not defined. This is probably
    # more flexible.
    def make_node(self, x, y):
        x = theano.tensor.as_tensor_variable(x)
        y = theano.tensor.as_tensor_variable(y)
        print('--- in make_node:')
        print('x:', type(x), repr(x))
        print('y:', type(y), repr(y))
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return theano.Apply(self, [x, y], [x.type(), y.type()])

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        print('--- in perform:')
        print('node:', repr(node))
        print('inputs_storage:', repr(inputs_storage))
        print('output_storage:', repr(output_storage))
        x = inputs_storage[0]
        y = inputs_storage[1]
        out1 = output_storage[0]
        out2 = output_storage[1]
        print('x:', type(x), repr(x))
        print('y:', type(y), repr(y))
        print('out1:', type(out1), repr(out1))
        # TODO: perform in-place
        out1[0] = np.asarray(self.odl_op(x))
        out2[0] = np.asarray(self.odl_op(y))

    # optional:
    check_input = True

    def grad(self, inputs, output_gradients):
        pass

    def R_op(self, inputs, eval_points):
        pass

    def infer_shape(self, node, input_shapes):
        print('--- in infer_shape:')
        print('node:', type(node), repr(node))
        print('input_shapes:', type(input_shapes), repr(input_shapes))
        return [self.odl_op.range.shape] * 2


# %% Testing

space = odl.rn(5)
ident = odl.ScalingOperator(space, 2)
theano_ident = TheanoOp(ident)

x = theano.tensor.dvector('x')
y = theano.tensor.dvector('y')
[z1, z2] = theano_ident(x, y)
print(repr(z1))
print(repr(z2))
f = theano.function([x, y], [z1, z2])
print(f(space.one(), space.one()))
