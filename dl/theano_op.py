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
    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        print('in make_node:')
        print('x:', repr(x))
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return theano.Apply(self, [x], [x.type()])

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        print('in perform:')
        print('node:', repr(node))
        print('inputs_storage:', repr(inputs_storage))
        print('output_storage:', repr(output_storage))
        x = inputs_storage[0]
        out = output_storage[0]
        print('x:', repr(x))
        print('out:', repr(out))
        out[0] = np.asarray(self.odl_op(x))

    # optional:
    check_input = True

    def grad(self, inputs, output_gradients):
        print('in grad:')
        print('inputs:', repr(inputs))
        print('output_gradients:', repr(output_gradients))
        x = inputs[0]
        z = output_gradients[0]
        print('x:', repr(x))
        print('z:', repr(z))
        # TODO: solution here? Check docs.
        return np.asarray(self.odl_op.derivative(x)(z))

    def R_op(self, inputs, eval_points):
        pass

    def infer_shape(self, node, input_shapes):
        print('in infer_shape:')
        print('node:', repr(node))
        print('input_shapes:', repr(input_shapes))
        return [self.odl_op.range.shape]


# %% Testing

space = odl.rn(5)
ident = odl.ScalingOperator(space, 2)
theano_ident = TheanoOp(ident)

x = theano.tensor.dvector('x')
z = theano_ident(x)
res = z.sum()
f = theano.function([x], res)
print(theano.pp(z))
print(f(space.one()))
g = theano.grad(res, x)
print(g)
