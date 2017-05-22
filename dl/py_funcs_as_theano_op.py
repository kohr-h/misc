import numpy as np
import theano


class CallableWrapperOp(theano.Op):
    """Wrap Python callables in an `theano.gof.op.Op`."""

    # Populate this tuple to tell __eq__ and __hash__ what to use for their
    # respective jobs
    __props__ = ('func', 'func_jac_adj')

    check_input = True

    def __init__(self, func, func_jac_adj=None, out_type=None,
                 infer_shape=None):
        """Initialize a new instance.

        Parameters
        ----------
        func : callable
            The callable object to be wrapped for forward evaluation.
            It must be a function of one array parameter and return one array.
        func_jac_adj : callable, optional
            The callable object to be wrapped for Jacobian adjoint evaluation.
            It must be a function of two array parameters of the same shape
            and return one array.
        out_type : theano.tensor.type.TensorVariable, optional
            Type of the output. The parameter can be eiter a variable
            type or a variable -- in both cases ``out_type.type()`` is
            used to infer the actual output type.
        infer_shape : callable, optional
            Function used to infer output shape from input shape.
            It must take a `theano.gof.graph.Apply` parameter ``node``
            and a (1-element) list of `theano.compile.ops.Shape`,
            and return a 1-element list of tuple or
            `theano.compile.ops.Shape`.

        Examples
        --------
        Wrapping a matrix multiplication including its Jacobian wrt the
        inputs:

        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype=float)

        >>> def f(x):
        ...     return matrix.dot(x) ** 2
        >>> def f_jac_adj(x, v):
        ...     return 2 * matrix.T.dot(matrix.dot(x) * v)
        >>> def infer_shape(node, input_shapes):
        ...     return [(matrix.shape[0],)]
        >>> op = CallableWrapperOp(f, f_jac_adj, infer_shape=infer_shape)
        >>> op.func is f
        True
        >>> op.func_jac_adj is f_jac_adj
        True

        The ``out_type`` parameter can be used as follows:

        >>> vector = np.array([1, 0, 1], dtype=float)
        >>> def f(x):
        ...     # Scalar output, hence this is a functional
        ...     return vector.dot(x) ** 2
        >>> out_var = theano.tensor.dscalar()
        >>> op = CallableWrapperOp(f, out_type=out_var)
        >>> op_apply = op.make_node(theano.tensor.dvector())
        >>> op_apply.outputs[0].type()
        <TensorType(float64, scalar)>
        """
        self.__func = func
        self.__func_jac_adj = func_jac_adj
        if out_type is None:
            self.__out_type = None
        elif isinstance(out_type, theano.tensor.TensorType):
            # For some reason we need to go this route since theano complains
            # otherwise (ownership something)
            self.__out_type = theano.tensor.TensorVariable(out_type).type()
        else:
            # TensorVariable was given
            self.__out_type = out_type.type()

        # If the optional arguments were not given, we don't create the
        # attributes.

        def _grad(inputs, output_grads):
            return self.R_op(inputs, output_grads)

        if self.func_jac_adj is not None:
            self.grad = _grad

        if infer_shape is not None:
            self.infer_shape = infer_shape

    @property
    def func(self):
        """The callable for forward evaluation wrapped in this Op."""
        return self.__func

    @property
    def func_jac_adj(self):
        """The callable for Jacobian adjoint evaluation wrapped in this Op.

        This is ``None`` if ``func_jac_adj`` was not provided during init.
        """
        return self.__func_jac_adj

    @property
    def out_type(self):
        """Output type of this Op's node in the computation graph.

        This is ``None`` if ``out_type`` was not provided during init.
        """
        return self.__out_type

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
        # TODO: map to scalar output in case we receive a functional.
        # Probably the best way to do this is to use an optional output
        # variable type in __init__.
        out_type = x.type() if self.out_type is None else self.out_type
        return theano.Apply(self, [x], [out_type])

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
            by the result of the application of `func`.

        Examples
        --------
        Perform matrix multiplication:

        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype=float)

        >>> def f(x):
        ...     return matrix.dot(x) ** 2
        >>> matrix_op = CallableWrapperOp(f)
        >>> x = theano.tensor.dvector('x')
        >>> op_x = matrix_op(x)
        >>> op_func = theano.function([x], [op_x])
        >>> op_func([1, 2, 3])
        [array([ 16.,  25.])]

        Evaluate the functional type operator:

        >>> vector = np.array([1, 0, 1], dtype=float)
        >>> def f(x):
        ...     # Scalar output, hence this is a functional
        ...     return vector.dot(x) ** 2
        >>> scalar = theano.tensor.dscalar()
        >>> dot_op = CallableWrapperOp(f, out_type=scalar)
        >>> op_x = dot_op(x)
        >>> op_func = theano.function([x], [op_x])
        >>> op_func([1, 2, 3])
        [array(16.0)]
        """
        x = inputs_storage[0]
        out = output_storage[0]
        # TODO: in-place (persistent output_storage)
        out[0] = np.asarray(self.func(x))

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
        Jacobian of the squared matrix product:

        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, 1, 1]], dtype=float)

        >>> def f(x):
        ...     return matrix.dot(x) ** 2
        >>> def f_jac_adj(x, v):
        ...     return 2 * matrix.T.dot(matrix.dot(x) * v)
        >>> theano_op = CallableWrapperOp(f, f_jac_adj)
        >>> x = theano.tensor.dvector('x')
        >>> op_x = theano_op(x)
        >>> cost = op_x.sum()
        >>> cost_grad = theano.grad(cost, x)
        >>> cost_grad_func = theano.function([x], [cost_grad])
        >>> cost_grad_func([1, 2, 3])
        [array([  8.,  10.,  18.])]

        Gradient of the functional:

        >>> vector = np.array([1, 0, 1], dtype=float)
        >>> def f(x):
        ...     # Scalar output, hence this is a functional
        ...     return vector.dot(x) ** 2
        >>> def f_jac_adj(x, v):
        ...     # This is now the gradient, multiplied by the scalar v
        ...     return 2 * vector.dot(x) * v * vector
        >>> out_var = theano.tensor.dscalar()
        >>> dot_op = CallableWrapperOp(f, f_jac_adj, out_type=out_var)
        >>> x = theano.tensor.dvector('x')
        >>> op_x = dot_op(x)
        >>> grad_dot_op = theano.grad(op_x, x)
        >>> grad_dot_func = theano.function([x], [grad_dot_op])
        >>> grad_dot_func([1, 2, 3])
        [array([ 8.,  0.,  8.])]

        Notes
        -----
        This method applies the contribution of this node, i.e., the Jacobian
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
            It uses a wrapper class ``CallableJacobianWrapperOp``
            for ``(x, v) --> op.derivative(x).adjoint(v)``.
        """
        x = inputs[0]
        pts = eval_points[0]

        op = self

        class CallableJacobianWrapperOp(theano.Op):

            """Wrap ``func_jac_adj`` into a Theano Op.

            This Op has two inputs, ``x`` and ``v``, where ``x``
            is the point at which the Jacobian is taken, and ``v`` the
            tensor to which it is applied. There is only one output,
            which is of the same type as ``v`` (and ``x``).
            """

            def make_node(self, x, v):
                """Create a node for the computation graph."""
                x = theano.tensor.as_tensor_variable(x)
                v = theano.tensor.as_tensor_variable(v)
                return theano.Apply(self, [x, v], [x.type()])

            def perform(self, node, inputs_storage, output_storage):
                """Evaluate this node's computation.

                This method computes ``func_jac_adj(x, v)``.
                """
                if op.func_jac_adj is None:
                    raise NotImplementedError(
                        'R_op undefined for `func_jac_adj=None`')
                x = inputs_storage[0]
                v = inputs_storage[1]
                # TODO: in-place (persistent output_storage)
                out = output_storage[0]
                out[0] = np.asarray(op.func_jac_adj(x, v))

        r_op = CallableJacobianWrapperOp()
        r_op_apply = r_op(x, pts)
        return [r_op_apply]

if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
