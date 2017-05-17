import numpy as np
import odl
import theano


class OdlOperatorAsTheanoOp(theano.Op):
    """Wrap an ODL operator in an `theano.gof.op.Op`.

    This variant implies that the gradient should be taken with respect to
    the input to the ODL operator. In other words, the ODL operator is
    assumed to be defined in terms of the learnable parameters.

    For instance, if the ODL operator implements a convolution with a kernel
    :math:`K`, and this kernel is supposed to be learned, then the ODL
    operator must be defined as :math:`K \mapsto K \\ast f` for fixed
    :math:`f`, not as an operator acting on :math:`f`.
    """

    # Populate this tuple to tell __eq__ and __hash__ what to use for their
    # respective jobs
    __props__ = ('odl_op',)

    check_input = True

    def __init__(self, odl_op):
        """Initialize a new instance.

        Parameters
        ----------
        odl_op : `Operator`
            The ODL operator to be wrapped. For gradient computations to
            work, ``odl_op.derivative`` must be implemented.

        Example
        -------
        >>> space = odl.rn(3)
        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype=float)
        >>> op = odl.MatrixOperator(matrix, domain=space)
        >>> theano_op = OdlOperatorAsTheanoOp(op)
        >>> theano_op.odl_op is op
        True
        """
        self.__odl_op = odl_op

    @property
    def odl_op(self):
        """The ODL `Operator` wrapped in this Op."""
        return self.__odl_op

    def make_node(self, x):
        """Create a node for the computation graph.

        Parameters
        ----------
        x : `theano.tensor.var.TensorVariable`
            Input to the node.

        Returns
        -------
        node : `theano.gof.graph.Apply`
            Node for the Theano expression graph. Its only input is ``x``,
            and the output is of the same type.
        """
        x = theano.tensor.as_tensor_variable(x)
        # TODO: we need to check that the dimension of the tensor variable
        # is correct. Alternatively we can reshape.
        # TODO: map to scalar output in case we receive a Functional
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs_storage, output_storage):
        """Evaluate this node's computation.

        Parameters
        ----------
        node : `theano.gof.graph.Apply`
            The node of this Op in the computation graph.
        inputs_storage : 1-element list of arrays
            Contains an array (usually `numpy.ndarray`) of concrete values
            supplied for the symbolic input variable ``x``.
        output_storage : 1-element list of 1-element lists
            The single 1-element list contained in ``output_storage``
            by default contains only ``None``. This value must be replaced
            by the result of the application of `odl_op`.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype=float)
        >>> op = odl.MatrixOperator(matrix, domain=space)
        >>> theano_op = OdlOperatorAsTheanoOp(op)
        >>> x = theano.tensor.dvector('x')
        >>> op_x = theano_op(x)
        >>> op_func = theano.function([x], [op_x])
        >>> op_func([1, 2, 3])
        [array([ 4.,  5.])]
        """
        x = inputs_storage[0]
        out = output_storage[0]
        # TODO: in-place (persistent output_storage)
        out[0] = np.asarray(self.odl_op(x))

    def grad(self, inputs, output_grads):
        """Apply the adjoint of the Jacobian at ``inputs`` to ``output_grads``.

        Parameters
        ----------
        inputs : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic input to the gradient, the point at which the
            Jacobian is computed.
        output_grads : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic gradient from the subsequent node received during
            backpropagation. The adjoint of the Jacobian is applied to
            this variable.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype=float)
        >>> op = odl.MatrixOperator(matrix, domain=space)
        >>> theano_op = OdlOperatorAsTheanoOp(op)
        >>> x = theano.tensor.dvector('x')
        >>> op_x = theano_op(x)
        >>> cost = op_x.sum()
        >>> cost_grad = theano.grad(cost, x)
        >>> cost_grad_func = theano.function([x], [cost_grad])
        >>> cost_grad_func([1, 2, 3])
        [array([ 1.,  1.,  2.])]
        >>> sum_grad = np.array([1.0, 1.0])
        >>> np.allclose(cost_grad_func([1, 2, 3]), matrix.T.dot(sum_grad))
        True

        Notes
        -----
        This method apply the contribution of this node, i.e., the Jacobian
        of its outputs with respect to its inputs, to the gradients of some
        cost function with respect to the outputs of this node.

        Example: Assume that this node computes :math:`x \mapsto f(x)`, and
        its output is connected to a cost function that computes
        :math:`y --> C(y)`. Here, :math:`x`, :math:`y` and :math:`f(x)` are
        tensor variables and :math:`C(y)` is a scalar variable.
        In ODL language, what ``grad`` should compute

            .. math::
                \\nabla(C \circ f)(x) = f'(x)^*\\big(\\nabla C (f(x))\\big)

        according to the chain rule. In ODL code, this corresponds to ::

            f.derivative(x).adjoint(C.gradient(f(x))).

        Then, the parameter ``output_grads`` contains a single tensor
        variable ``y`` that stands for :math:`\\nabla C(f(x))`. Thus,
        ``grad`` boils down to taking the ``output_grads`` ``[y]`` and
        return ``[f'(x)^*(y)]`` symbolically, where ``inputs == [x]``.

        This turns out to be just a special case of `R_op`, which is the
        exact same operation, only for arbitrary ``eval_points`` instead of
        ``output_grads``.
        """
        return self.R_op(inputs, output_grads)

    def R_op(self, inputs, eval_points):
        """Apply the adjoint of the Jacobian at ``inputs`` to ``eval_points``.

        This is the symbolic counterpart of ODL's ::

            op.derivative(x).adjoint(v)

        See `grad` for its usage.

        Parameters
        ----------
        inputs : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic input to the gradient, the point at which the
            Jacobian is computed.
        eval_points : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic input to the adjoint of the Jacobian, i.e., the
            variable to which the Jacobian adjoint should be applied.

        Returns
        -------
        outputs : 1-element list of `theano.tensor.var.TensorVariable`
            Symbolic result of the application of the Jacobian adjoint.
            It uses a wrapper class ``OdlDerivativeAdjointAsTheanoROp``
            for ``(x, v) --> op.derivative(x).adjoint(v)``.
        """
        x = inputs[0]
        pts = eval_points[0]

        op = self

        class OdlDerivativeAdjointAsTheanoROp(theano.Op):

            """Wrap ``op.derivative`` into a Theano Op.

            This Op has two inputs, ``x`` and ``v``, where ``x``
            is the point at which the Jacobian is taken, and ``v`` the
            tensor to which it is applied. There is only one output,
            which is of the same type as ``v`` (and ``x``).
            """

            def make_node(self, x, v):
                """Create a node for the computation graph."""
                x = theano.tensor.as_tensor_variable(x)
                v = theano.tensor.as_tensor_variable(v)
                return theano.Apply(self, [x, v], [v.type()])

            def perform(self, node, inputs_storage, output_storage):
                """Evaluate this node's computation.

                This method computes ::

                    op.derivative(x).adjoint(v)
                """
                x = inputs_storage[0]
                v = inputs_storage[1]
                # TODO: in-place (persistent output_storage)
                out = output_storage[0]
                out[0] = np.asarray(op.odl_op.derivative(x).adjoint(v))

            def infer_shape(self, node, input_shapes):
                return [op.odl_op.domain.shape]

        r_op = OdlDerivativeAdjointAsTheanoROp()
        r_op_apply = r_op(x, pts)
        return [r_op_apply]

    def infer_shape(self, node, input_shapes):
        """Return a list of output shapes based on ``input_shapes``.

        This method is optional. It allows to compute the shape of the
        output without having to evaluate.

        Parameters
        ----------
        node : `theano.gof.graph.Apply`
            The node of this Op in the computation graph.
        input_shapes : 1-element list of `theano.compile.ops.Shape`
            Symbolic shape of the input.

        Returns
        -------
        output_shapes : 1-element list of tuples
            Fixed shape of the output determined by `odl_op`.
        """
        return [self.odl_op.range.shape]

if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
