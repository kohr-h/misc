import odl
import numpy as np
import pygpu

rn = odl.rn((3, 4), impl='gpuarray')

x = rn.element(np.arange(rn.size).reshape(rn.shape))
y = rn.one()
z = 2 * rn.one()

slc = [slice(None)] * (rn.ndim - 1) + [slice(None, None, 2)]
res_space = rn.element()[slc].space

x, y, z = x[slc], y[slc], z[slc]
print('x')
print(x)
print(x.data.flags)
print('')
print('y')
print(y)
print(y.data.flags)
print('')
print('z')
print(z)
print(z.data.flags)
print('')


def ielemwise3(y, op, a, x, oper=None, op_tmpl="y = a {op} x",
               broadcast=False, convert_f16=True):
    if not isinstance(a, pygpu.gpuarray.GpuArray):
        a = np.asarray(a)
    if not isinstance(x, pygpu.gpuarray.GpuArray):
        x = np.asarray(x)

    y_arg = pygpu.elemwise.as_argument(y, 'y', read=True, write=True)
    a_arg = pygpu.elemwise.as_argument(a, 'a', read=True)
    x_arg = pygpu.elemwise.as_argument(x, 'x', read=True)

    args = [y_arg, a_arg, x_arg]

    if oper is None:
        oper = op_tmpl.format(op=op)

    k = pygpu.elemwise.GpuElemwise(y.context, oper, args,
                                   convert_f16=convert_f16)
    k(y, a, x, broadcast=broadcast)
    return y


def ielemwise4(z, a, op1, x, op2, y, oper=None,
               op_tmpl="z = a {op1} x {op2} y",
               broadcast=False, convert_f16=True):
    if not isinstance(a, pygpu.gpuarray.GpuArray):
        a = np.asarray(a)
    if not isinstance(x, pygpu.gpuarray.GpuArray):
        x = np.asarray(x)
    if not isinstance(y, pygpu.gpuarray.GpuArray):
        y = np.asarray(y)

    z_arg = pygpu.elemwise.as_argument(z, 'z', write=True)
    a_arg = pygpu.elemwise.as_argument(a, 'a', read=True)
    x_arg = pygpu.elemwise.as_argument(x, 'x', read=True)
    y_arg = pygpu.elemwise.as_argument(y, 'y', read=True)

    args = [z_arg, a_arg, x_arg, y_arg]

    if oper is None:
        oper = op_tmpl.format(op1=op1, op2=op2)

    print(oper)

    k = pygpu.elemwise.GpuElemwise(z.context, oper, args,
                                   convert_f16=convert_f16)
    k(z, a, x, y, broadcast=broadcast)
    return y


def ielemwise5(z, a, op1, x, op2, b, op3, y, oper=None,
               op_tmpl="z = a {op1} x {op2} b {op3} y",
               broadcast=False, convert_f16=True):
    if not isinstance(a, pygpu.gpuarray.GpuArray):
        a = np.asarray(a)
    if not isinstance(x, pygpu.gpuarray.GpuArray):
        x = np.asarray(x)
    if not isinstance(b, pygpu.gpuarray.GpuArray):
        b = np.asarray(b)
    if not isinstance(y, pygpu.gpuarray.GpuArray):
        y = np.asarray(y)

    z_arg = pygpu.elemwise.as_argument(z, 'z', write=True)
    a_arg = pygpu.elemwise.as_argument(a, 'a', read=True)
    x_arg = pygpu.elemwise.as_argument(x, 'x', read=True)
    b_arg = pygpu.elemwise.as_argument(b, 'b', read=True)
    y_arg = pygpu.elemwise.as_argument(y, 'y', read=True)

    args = [z_arg, a_arg, x_arg, b_arg, y_arg]

    if oper is None:
        oper = op_tmpl.format(op1=op1, op2=op2, op3=op3)

    print(oper)

    k = pygpu.elemwise.GpuElemwise(z.context, oper, args,
                                   convert_f16=convert_f16)
    k(z, a, x, b, y, broadcast=broadcast)
    return y


print('------ test -------')
a = 3.14
b = 3

# pygpu.elemwise.ielemwise2(x.data, '+', 3.14)
# odl.space.gpuary_tensors.scal(b, y.data, x.data)
# ielemwise3(z.data, '*', a, x.data)
ielemwise4(z.data, a, '*', x.data, '+', y.data)
# ielemwise5(z.data, a, '*', x.data, '+', b, '*', y.data)
print('')
print(z.data)
# res_space.lincomb(a, x, b, y, out=x)
