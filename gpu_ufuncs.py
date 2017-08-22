#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:25:39 2017

@author: kohr
"""
from pygpu._elemwise import arg
from pygpu.dtypes import dtype_to_ctype
from pygpu.elemwise import as_argument, GpuElemwise
from pygpu.gpuarray import GpuArray, array, empty, get_default_context

import mako
import numpy
from pkg_resources import parse_version
import warnings

# TODO: make explicit list of ufuncs
if parse_version(numpy.__version__) < parse_version('1.13.0'):
    raise RuntimeError('Numpy version 1.13 reqiured')

# Not supported:
# - 'e' (float16)  -- not yet
# - 'FDG' (complex64/128/256)  -- not yet
# - 'SU' (str/unicode)
# - 'V' (void)
# - 'O' (object)
# - 'Mm' (datetime)
TYPECODES = '?bhilqpBHILQPfdg'
TYPECODES_TO_DTYPES = {tc: numpy.dtype(tc) for tc in TYPECODES}


# Collect all Numpy ufuncs
ufuncs = []
for name in dir(numpy):
    obj = getattr(numpy, name)
    if isinstance(obj, numpy.ufunc) and name == obj.__name__:
        ufuncs.append(obj)


# Initialize metadata dict from Numpy by populating the basic properties
ufunc_metadata = {}
for ufunc in ufuncs:
    entry = {}

    # Defined by numpy
    entry['nin'] = ufunc.nin
    entry['nout'] = ufunc.nout
    entry['nargs'] = ufunc.nargs
    entry['identity'] = ufunc.identity
    entry['types'] = ufunc.types
    entry['ntypes'] = ufunc.ntypes

    # Docstring
    entry['doc'] = ''

    # Alternative names, for duplication in module namespace
    entry['alt_names'] = ()

    # Valid range of inputs written as, e.g., '[-1, 1]',
    # '[-inf, -1) + (1, inf]' or 'R x [0, 1]'. The '+' symbol can be used
    # for set unions, and an 'x' to separate domains of input variables.
    # The letters 'R', 'N' and 'Z' can be used for the real, natural
    # and integer numbers, respectively.
    # This is only for testing purposes to determine valid input.
    entry['domain'] = ''

    # Numpy version that added the ufunc
    entry['npy_ver_added'] = ''

    # The following entries implement the ufunc in C. If all 3 possible
    # variants 'c_func', 'c_op' and 'oper_fmt' are `None`, the ufunc is
    # is considered not implemented.

    # Corresponding C function name if existing, else None
    entry['c_func'] = None

    # Corresponding C operator symbol if existing, else None
    entry['c_op'] = None

    # Format string for `oper` string in `GpuElementwise`, necessary
    # only if both 'c_func' and 'c_op' are `None`.
    #
    # It will be used as follows:
    #
    #    Unary functions:
    #        oper = oper_fmt.format(idt=c_cast_dtype_in,
    #                               odt=c_cast_dtype_out)
    #
    #    Unary functions with 2 outputs:
    #        oper = oper_fmt.format(idt=c_cast_dtype_in,
    #                               odt1=c_cast_dtype_out1,
    #                               odt2=c_cast_dtype_out2)
    #
    #    Binary functions:
    #        oper = oper_fmt.format(idt1=c_cast_dtype_in1,
    #                               idt2=c_cast_dtype_in2,
    #                               odt=c_cast_dtype_out)
    #
    #    Binary functions with 2 outputs:
    #        oper = oper_fmt.format(idt1=c_cast_dtype_in1,
    #                               idt2=c_cast_dtype_in2,
    #                               odt1=c_cast_dtype_out1,
    #                               odt2=c_cast_dtype_out2)
    #
    # Here, `c_cast_dtype_*` are strings of C dtypes used for casting
    # of input and output data types, e.g., for `true_divide`:
    #
    #     oper_fmt = 'out = ({odt}) (({idt1}) a / ({idt2}) b)'
    #
    entry['oper_fmt'] = None

    # Mako template string for a preamble to be used as `preamble`
    # parameter in `GpuElementwise`. Only used together with 'oper_fmt'.
    #
    # It will be used as follows:
    #
    #     template = mako.template.Template(oper_preamble_tpl)
    #     preamble = template.render(...)
    #
    # The arguments to `template.render()` are the same as to
    # `oper_fmt.format()`, with an additional `idtmax` for the
    # maximum of the input dtypes (for ufuncs with 2 arguments) that
    # is used for some intermediate computations.
    #
    entry['oper_preamble_tpl'] = ''

    # Dictionary to document the incompatibilities with Numpy, not
    # including unsupported data types. This is to skip tests that would
    # otherwise fail.
    # Each entry of this dictionary is itself a  dictionary with the
    # following possible items:
    #
    #     - 'dtypes' : tuple of (dtype, tuple or None)
    #           Data types for which an incompatibility occurs.
    #           The tuple contains `nin` entries, corresponding to
    #           the inputs. A `None` entry means "all dtypes for
    #           this input", a tuple of dtypes means "all the dtypes
    #           in the tuple".
    #     - 'op_type' : string or None
    #           Operand type that triggers an incompatibility, e.g.,
    #           'negative_scalar'.
    #     - 'reason' : string
    #           An explanation of the incompatibility.
    #
    # Present keys (except 'reason') are AND'ed together, i.e., when
    # checking whether to skip a test, all conditions must be met.
    #
    # The dictionaries in the top-level dictionary are OR'ed together.
    #
    entry['npy_incompat'] = {}

    ufunc_metadata[ufunc.__name__] = entry


def patch_types(types):
    """Return a new list with unsupported type signatures removed."""
    new_types = []
    for sig in types:
        tc_in, tc_out = sig.split('->')
        if not (all(c in TYPECODES for c in tc_in) and
                all(c in TYPECODES for c in tc_out)):
            # Signature contains unsupported type, not adding
            continue
        else:
            new_types.append(sig)
    return new_types


for meta in ufunc_metadata.values():
    meta['types'] = patch_types(meta['types'])
    meta['ntypes'] = len(meta['types'])


# %% Set the individual metadata entries

# TODO: add docstrings

# --- absolute --- #

ufunc_metadata['absolute']['alt_names'] = ('abs',)
# The C function can be abs or fabs, needs special treatment
ufunc_metadata['absolute']['c_func'] = ''

# --- add --- #

ufunc_metadata['add']['c_op'] = '+'

# --- arccos --- #

ufunc_metadata['arccos']['c_func'] = 'acos'
ufunc_metadata['arccos']['domain'] = '[-1, 1]'

# --- arccosh --- #

ufunc_metadata['arccosh']['c_func'] = 'acosh'
ufunc_metadata['arccosh']['domain'] = '[1, inf]'

# --- arcsin --- #

ufunc_metadata['arcsin']['c_func'] = 'asin'
ufunc_metadata['arcsin']['domain'] = '[-1, 1]'

# --- arcsinh --- #

ufunc_metadata['arcsinh']['c_func'] = 'asinh'

# --- arctan --- #

ufunc_metadata['arctan']['c_func'] = 'atan'

# --- arctan2 --- #

ufunc_metadata['arctan2']['c_func'] = 'atan2'

# --- arctanh --- #

ufunc_metadata['arctanh']['c_func'] = 'atanh'
ufunc_metadata['arctanh']['domain'] = '[-1, 1]'

# --- bitwise_and --- #

ufunc_metadata['bitwise_and']['c_op'] = '&'

# --- bitwise_or --- #

ufunc_metadata['bitwise_or']['c_op'] = '|'

# --- bitwise_xor --- #

ufunc_metadata['bitwise_xor']['c_op'] = '^'

# --- cbrt --- #

ufunc_metadata['cbrt']['npy_ver_added'] = '1.10.0'
ufunc_metadata['cbrt']['c_func'] = 'cbrt'

# --- ceil --- #

ufunc_metadata['ceil']['c_func'] = 'ceil'

# --- conjugate --- #

# Leave unimplemented by not adding anything

# --- copysign --- #

ufunc_metadata['copysign']['c_func'] = 'copysign'

# --- cos --- #

ufunc_metadata['cos']['c_func'] = 'cos'

# --- cosh --- #

ufunc_metadata['cosh']['c_func'] = 'cosh'

# --- deg2rad --- #

_oper_fmt = 'out = ({{odt}})({:.45f} * ({{idt}}) a)'.format(numpy.deg2rad(1.0))
ufunc_metadata['deg2rad']['oper_fmt'] = _oper_fmt

# --- degrees --- #

_oper_fmt = 'out = ({{odt}})({:.45f} * ({{idt}}) a)'.format(numpy.degrees(1.0))
ufunc_metadata['degrees']['oper_fmt'] = _oper_fmt

# --- divmod --- #

_preamble_tpl = '''
WITHIN_KERNEL ${odt1}
divmod(${idt1} a, ${idt2} b, ${odt2} *out2) {
    if (b == 0) {
        *out2 = 0;
        return 0;
    }
    ${idtmax} quot = (${idtmax}) a / b;
    *out2 = (${odt2}) quot;
    return a - quot * b;
}
'''
ufunc_metadata['divmod']['c_func'] = 'divmod'
ufunc_metadata['divmod']['oper_preamble_tpl'] = _preamble_tpl
ufunc_metadata['divmod']['npy_ver_added'] = '1.13.0'

# --- equal --- #

ufunc_metadata['equal']['c_op'] = '=='

# --- exp --- #

ufunc_metadata['exp']['c_func'] = 'exp'

# --- exp2 --- #

ufunc_metadata['exp2']['c_func'] = 'exp2'

# --- expm1 --- #

ufunc_metadata['expm1']['c_func'] = 'expm1'

# --- fabs --- #

ufunc_metadata['fabs']['c_func'] = 'fabs'

# --- float_power --- #

ufunc_metadata['float_power']['c_func'] = 'pow'
ufunc_metadata['float_power']['npy_ver_added'] = '1.12.0'

# --- floor --- #

ufunc_metadata['floor']['c_func'] = 'floor'

# --- floor_divide --- #

# implement as sign(a/b) * int(abs(a/b) + shift(a,b))
# where shift(a,b) = 0 if sign(a) == sign(b) else 1 - epsilon
_preamble_tpl = '''
WITHIN_KERNEL ${odt}
floor_divide(${idt1} a, ${idt2} b) {
    ${idtmax} quot = (${idtmax}) a / b;
    if ((a < 0) != (b < 0)) {
        return (${odt}) -(quot + 0.999);
    } else {
        return (${odt}) quot;
    }
}
'''
ufunc_metadata['floor_divide']['c_func'] = 'floor_divide'
ufunc_metadata['floor_divide']['oper_preamble_tpl'] = _preamble_tpl

# --- fmax --- #

# Same as `maximum`, but different handling of NaNs
_preamble_tpl = '''
WITHIN_KERNEL ${odt}
fmax(${idt1} a, ${idt2} b) {
    if isnan(a) {
        return b;
    }
    else if isnan(b) {
        return a;
    }
    else {
        return (a > b) ? a : b;
    }
}
'''
ufunc_metadata['fmax']['c_func'] = 'fmax'
ufunc_metadata['fmax']['oper_preamble_tpl'] = _preamble_tpl

# --- fmin --- #

# Same as `minimum`, but different handling of NaNs
_preamble_tpl = '''
WITHIN_KERNEL ${odt}
fmin(${idt1} a, ${idt2} b) {
    if isnan(a) {
        return b;
    }
    else if isnan(b) {
        return a;
    }
    else {
        return (a > b) ? a : b;
    }
}
'''
ufunc_metadata['fmin']['c_func'] = 'fmin'
ufunc_metadata['fmin']['oper_preamble_tpl'] = _preamble_tpl

# --- fmod --- #

ufunc_metadata['fmod']['c_func'] = 'fmod'

# --- frexp --- #

ufunc_metadata['frexp']['c_func'] = 'frexp'

# --- greater --- #

ufunc_metadata['greater']['c_op'] = '>'

# --- greater_equal --- #

ufunc_metadata['greater_equal']['c_op'] = '>='

# --- heaviside --- #

_preamble_tpl = '''
WITHIN_KERNEL ${odt}
heaviside(${idt1} a, ${idt2} b) {
    if (a < 0) {
        return 0;
    }
    else if (a == 0) {
        return b;
    }
    else {
        return 1;
    }
}
'''
ufunc_metadata['heaviside']['c_func'] = 'heaviside'
ufunc_metadata['heaviside']['oper_preamble_tpl'] = _preamble_tpl
ufunc_metadata['heaviside']['npy_ver_added'] = '1.13.0'

# --- hypot --- #

ufunc_metadata['hypot']['c_func'] = 'hypot'

# --- invert --- #

ufunc_metadata['invert']['c_op'] = '~'
ufunc_metadata['invert']['alt_names'] = ('bitwise_not',)

# --- isfinite --- #

_oper_fmt = 'out = ({odt}) (a != INFINITY && a != -INFINITY && !isnan(a))'
ufunc_metadata['isfinite']['oper_fmt'] = _oper_fmt

# --- isinf --- #

_oper_fmt = 'out = ({odt}) (a == INFINITY || a == -INFINITY)'
ufunc_metadata['isinf']['oper_fmt'] = _oper_fmt

# --- isnan --- #

_oper_fmt = 'out = ({odt}) (abs(isnan(a)))'
ufunc_metadata['isnan']['oper_fmt'] = _oper_fmt

# --- ldexp --- #

ufunc_metadata['ldexp']['c_func'] = 'ldexp'

# --- left_shift --- #

ufunc_metadata['left_shift']['c_op'] = '<<'

# --- less --- #

ufunc_metadata['less']['c_op'] = '<'

# --- less_equal --- #

ufunc_metadata['less_equal']['c_op'] = '<='

# --- log --- #

ufunc_metadata['log']['c_func'] = 'log'
ufunc_metadata['log']['domain'] = '(0, inf]'

# --- log10 --- #

ufunc_metadata['log10']['c_func'] = 'log10'
ufunc_metadata['log10']['domain'] = '(0, inf]'

# --- log1p --- #

ufunc_metadata['log1p']['c_func'] = 'log1p'
ufunc_metadata['log1p']['domain'] = '(0, inf]'

# --- log2 --- #

ufunc_metadata['log2']['c_func'] = 'log2'
ufunc_metadata['log2']['domain'] = '(0, inf]'

# --- logaddexp --- #

_oper_fmt = 'out = ({odt}) log(exp(a) + exp(b))'
ufunc_metadata['logaddexp']['oper_fmt'] = _oper_fmt

# --- logaddexp2 --- #

_oper_fmt = '''
out = ({odt}) log(exp(a * log(2.0)) + exp(b * log(2.0))) / log(2.0)
'''
ufunc_metadata['logaddexp2']['oper_fmt'] = _oper_fmt

# --- logical_and --- #

ufunc_metadata['logical_and']['c_op'] = '&&'

# --- logical_not --- #

ufunc_metadata['logical_not']['c_op'] = '!'

# --- logical_or --- #

ufunc_metadata['logical_or']['c_op'] = '||'

# --- logical_xor --- #

_oper_fmt = 'out = ({odt}) (a ? !b : b)'
ufunc_metadata['logical_xor']['oper_fmt'] = _oper_fmt

# --- maximum --- #

_oper_fmt = 'out = ({odt}) ((a > b) ? a : b)'
ufunc_metadata['maximum']['oper_fmt'] = _oper_fmt

# --- minimum --- #

_oper_fmt = 'out = ({odt}) ((a < b) ? a : b)'
ufunc_metadata['minimum']['oper_fmt'] = _oper_fmt

# --- modf --- #

ufunc_metadata['modf']['c_func'] = 'modf'

# --- multiply --- #

ufunc_metadata['multiply']['c_op'] = '*'

# --- negative --- #

ufunc_metadata['negative']['c_op'] = '-'

# --- nextafter --- #

ufunc_metadata['nextafter']['c_func'] = 'nextafter'

# --- not_equal --- #

ufunc_metadata['not_equal']['c_op'] = '!='

# --- positive --- #

ufunc_metadata['positive']['c_op'] = '+'
ufunc_metadata['positive']['npy_ver_added'] = '1.13.0'

# --- power --- #

# Integer to negative integer power raises ValueError in Numpy, too
# complicated to encode in 'domain'
ufunc_metadata['power']['c_func'] = 'pow'

# --- rad2deg --- #

_oper_fmt = 'out = ({{odt}})({:.45f} * ({{idt}}) a)'.format(numpy.degrees(1.0))
ufunc_metadata['rad2deg']['oper_fmt'] = _oper_fmt

# --- radians --- #

_oper_fmt = 'out = ({{odt}})({:.45f} * ({{idt}}) a)'.format(numpy.deg2rad(1.0))
ufunc_metadata['radians']['oper_fmt'] = _oper_fmt

# --- reciprocal --- #

_oper_fmt = 'out = ({odt}) (({odt}) 1.0) / a'
ufunc_metadata['reciprocal']['oper_fmt'] = _oper_fmt

# --- remainder --- #

_preamble_tpl = '''
WITHIN_KERNEL ${odt}
remainder(${idt1} a, ${idt2} b) {
    if (b == 0) {
        return 0;
    }
    ${idtmax} quot = (${idtmax}) a / b;
    return a - quot * b;
}
'''
ufunc_metadata['remainder']['c_func'] = 'remainder'
ufunc_metadata['remainder']['oper_preamble_tpl'] = _preamble_tpl
ufunc_metadata['remainder']['alt_names'] = ('mod',)

# --- right_shift --- #

ufunc_metadata['right_shift']['c_op'] = '>>'

# --- rint --- #

ufunc_metadata['rint']['c_func'] = 'rint'

# --- sign --- #

_oper_fmt = 'out = ({odt}) ((a > 0) ? 1 : (a < 0) ? -1 : 0)'
ufunc_metadata['sign']['oper_fmt'] = _oper_fmt

# --- signbit --- #

_oper_fmt = 'out = ({odt}) (a < 0)'
ufunc_metadata['signbit']['oper_fmt'] = _oper_fmt

# --- sin --- #

ufunc_metadata['sin']['c_func'] = 'sin'

# --- sinh --- #

ufunc_metadata['sinh']['c_func'] = 'sinh'

# --- spacing --- #

_oper_fmt = '''
out = ({odt}) ((a < 0) ?
               nextafter(a, ({idt}) a - 1) - a :
               nextafter(a, ({idt}) a + 1) - a)
'''
ufunc_metadata['spacing']['oper_fmt'] = _oper_fmt

# --- sqrt --- #

ufunc_metadata['sqrt']['c_func'] = 'sqrt'
ufunc_metadata['sqrt']['domain'] = '[0, inf]'

# --- square --- #

_oper_fmt = 'out = ({odt}) (a * a)'
ufunc_metadata['square']['oper_fmt'] = _oper_fmt

# --- subtract --- #

ufunc_metadata['subtract']['c_op'] = '-'

# --- tan --- #

ufunc_metadata['tan']['c_func'] = 'tan'

# --- tanh --- #

ufunc_metadata['tanh']['c_func'] = 'tanh'

# --- true_divide --- #

_oper_fmt = 'out = ({odt}) (({idt1}) a / ({idt2}) b)'
ufunc_metadata['true_divide']['oper_fmt'] = _oper_fmt
ufunc_metadata['true_divide']['alt_names'] = ('divide',)

# --- trunc --- #

ufunc_metadata['trunc']['c_func'] = 'trunc'

_oper_fmt = None
_preamble_tpl = None

# %%


def find_smallest_valid_signature(ufunc_name, inputs, outputs):
    """Return the smallest signature that can handle in & out dtypes.

    Parameters
    ----------
    ufunc_name : str
        Name of the ufunc for which the signature should be determined.
    inputs : sequence
        List of input arrays. Its length must be equal to the number of
        input arguments to the ufunc.
    outputs : sequence
        List of output arrays or ``None``. Its length must be equal to the
        number of output arguments of the ufunc. A ``None`` entry in the
        sequence will be ignored in the signature comparison.

    Returns
    -------
    signature : str
        Signature string of the form ``'[from]->[to]'``, where ``[from]``
        and ``[to]`` are strings of length ``nin`` and ``nout``, resp.,
        each character representing a typecode.

        Example: ``fi->f`` for ``('float32', 'int32') -> 'float32'``.
    """
    meta = ufunc_metadata[ufunc_name]
    assert len(inputs) == meta['nin']
    assert len(outputs) == meta['nout']

    types = meta['types']
    dtypes_in = [inp.dtype for inp in inputs]
    dtypes_out = [None if out is None else out.dtype for out in outputs]

    def supports_in_out_dtypes(sig):
        """Filter for signatures that support our current in & out dtypes."""
        from_part, to_part = sig.split('->')
        dtypes_from = tuple(numpy.dtype(c) for c in from_part)
        dtypes_to = tuple(numpy.dtype(c) for c in to_part)
        left_ok = all(dt >= dt_in
                      for dt, dt_in in zip(dtypes_from, dtypes_in))
        right_ok = all(dt >= dt_out
                       for dt, dt_out in zip(dtypes_to, dtypes_out)
                       if dt_out is not None)
        return left_ok and right_ok

    valid_sigs = filter(supports_in_out_dtypes, types)

    def dtypes_in_key(sig):
        """Key function for signature comparison according to input dtypes.

        It results in comparison of all typecodes on the left side of the
        signature since they are assembled in a tuple.
        """
        from_part = sig.split('->')[0]
        return tuple(numpy.dtype(c) for c in from_part)

    try:
        return min(valid_sigs, key=dtypes_in_key)
    except ValueError:
        return ''


def ufunc11(name, a, out=None, context=None):
    """Call a ufunc with 1 input and 1 output.

    Parameters
    ----------
    name : str
        Name of the NumPy ufunc.
    a : `array-like`
        Input array to which the ufunc should be applied.
    out : `pygpu.gpuarray.GpuArray`, optional
        Array in which to store the result.
    context : `pygpu.gpuarray.GpuContext`, optional
        Use this GPU context to evaluate the GPU kernel. For ``None``,
        if no GPU array is among the provided parameters, a default
        GPU context must have been set.

    Returns
    -------
    out : `pygpu.gpuarray.GpuArray`
        Result of the computation. If ``out`` was given, the returned
        object is a reference to it.
        The type of the returned array is `pygpu._array.ndgpuarray` if

        - no GPU array was among the parameters or
        - one of the parameters had type `pygpu._array.ndgpuarray`.
    """
    # Lazy import to avoid circular dependency
    from pygpu._array import ndgpuarray

    # --- Prepare input array --- #

    # Determine GPU context and class. Use the "highest" class present in the
    # inputs, defaulting to `ndgpuarray`
    need_context = True
    cls = None
    for ary in (a, out):
        if isinstance(ary, GpuArray):
            if context is not None and ary.context != context:
                raise ValueError('cannot mix contexts')
            context = ary.context
            if cls is None or cls == GpuArray:
                cls = ary.__class__
            need_context = False

    if need_context and context is None:
        context = get_default_context()
        cls = ndgpuarray

    # Cast input to `GpuArray` of the right dtype if necessary
    if isinstance(a, (GpuArray, numpy.ndarray)):
        if a.flags.f_contiguous and not a.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'

        # Determine signature here to avoid creating an intermediate GPU array
        sig = find_smallest_valid_signature(name, (a,), (out,))
        if not sig:
            raise TypeError('ufunc {!r} not supported for the input types, '
                            'and the inputs could not be safely coerced'
                            ''.format(name))

        tc_in, _ = sig.split('->')
        a = array(a, dtype=tc_in, copy=False, order=order, context=context,
                  cls=cls)
    else:
        a = array(a, context=context, cls=cls)

        sig = find_smallest_valid_signature(name, (a,), (out,))
        if not sig:
            raise TypeError('ufunc {!r} not supported for the input types, '
                            'and the inputs could not be safely coerced'
                            ''.format(name))

        # Upcast input if necessary
        tc_in, tc_out = sig.split('->')
        if a.dtype < tc_in:
            a = a.astype(tc_in)

    # Create output array if not provided
    if out is None:
        out = empty(a.shape, dtype=tc_out, context=context, cls=cls)

    # --- Generate code strings for GpuElemwise --- #

    # C dtypes for casting
    c_dtype_in = dtype_to_ctype(tc_in)
    c_dtype_out = dtype_to_ctype(tc_out)

    meta = ufunc_metadata[name]
    assert meta['nin'] == 1
    assert meta['nout'] == 1

    # Create `oper` string
    if meta['c_op'] is not None:
        # Case 1: unary operator
        unop = meta['c_op']
        if a.dtype == numpy.bool and unop == '-':
            if parse_version(numpy.__version__) >= parse_version('1.13'):
                # Numpy >= 1.13 raises a TypeError
                raise TypeError(
                    'negation of boolean arrays is not supported, use '
                    '`logical_not` instead')
            else:
                # Warn and remap to logical not
                warnings.warn('using negation (`-`) with boolean arrays is '
                              'deprecated, use `logical_not` (`~`) instead; '
                              'the current behavior will be changed along '
                              "with NumPy's", FutureWarning)
                unop = '!'
        oper = 'out = ({odt}) {}a'.format(unop, odt=c_dtype_out)
        preamble = ''

    elif meta['c_func'] is not None:
        # Case 2: C function
        c_func = meta['c_func']

        if name in ('abs', 'absolute'):
            # Special case
            if numpy.dtype(tc_out).kind == 'u':
                # Shortcut for abs() with unsigned int. This also fixes a CUDA
                # quirk that makes abs() crash with unsigned int input.
                out[:] = a
                return out
            elif numpy.dtype(tc_out).kind == 'f':
                c_func = 'fabs'
            else:
                c_func = 'abs'

        oper = 'out = ({odt}) {}(a)'.format(c_func, odt=c_dtype_out)
        preamble_tpl = mako.template.Template(meta['oper_preamble_tpl'])
        preamble = preamble_tpl.render(idt=c_dtype_in, odt=c_dtype_out)

    elif meta['oper_fmt'] is not None:
        # Case 3: custom implementation with `oper` template
        oper = meta['oper_fmt'].format(idt=c_dtype_in, odt=c_dtype_out)
        preamble_tpl = mako.template.Template(meta['oper_preamble_tpl'])
        preamble = preamble_tpl.render(idt=c_dtype_in, odt=c_dtype_out)

    else:
        # Case 4: not implemented
        raise NotImplementedError('ufunc {!r} not implemented'.format(name))

    # --- Generate and run GpuElemwise kernel --- #

    a_arg = as_argument(a, 'a', read=True)
    args = [arg('out', out.dtype, write=True), a_arg]

    ker = GpuElemwise(context, oper, args, preamble=preamble)
    ker(out, a)
    return out


def ufunc21(name, a, b, out=None, context=None):
    """Call a ufunc with 2 inputs and 1 output.

    Parameters
    ----------
    name : str
        Name of the NumPy ufunc.
    a, b : `array-like`
        Input arrays to which the ufunc should be applied.
    out : `pygpu.gpuarray.GpuArray`, optional
        Array in which to store the result.
    context : `pygpu.gpuarray.GpuContext`, optional
        Use this GPU context to evaluate the GPU kernel. For ``None``,
        if no GPU array is among the provided parameters, a default
        GPU context must have been set.

    Returns
    -------
    out : `pygpu.gpuarray.GpuArray`
        Result of the computation. If ``out`` was given, the returned
        object is a reference to it.
        The type of the returned array is `pygpu._array.ndgpuarray` if

        - no GPU array was among the parameters or
        - one of the parameters had type `pygpu._array.ndgpuarray`.
    """
    # Lazy import to avoid circular dependency
    from pygpu._array import ndgpuarray

    # --- Prepare input array --- #

    # Determine GPU context and class. Use the "highest" class present in the
    # inputs, defaulting to `ndgpuarray`
    need_context = True
    cls = None
    for ary in (a, b, out):
        if isinstance(ary, GpuArray):
            if context is not None and ary.context != context:
                raise ValueError('cannot mix contexts')
            context = ary.context
            if cls is None or cls == GpuArray:
                cls = ary.__class__
            need_context = False

    if need_context and context is None:
        context = get_default_context()
        cls = ndgpuarray

    # Cast input to `GpuArray` of the right dtype if necessary
    # TODO: figure out what to do here exactly (scalars and such)
    if isinstance(a, (GpuArray, numpy.ndarray)):
        if a.flags.f_contiguous and not a.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'

        # Determine signature here to avoid creating an intermediate GPU array
        sig = find_smallest_valid_signature(name, (a,), (out,))
        if not sig:
            raise TypeError('ufunc {!r} not supported for the input types, '
                            'and the inputs could not be safely coerced'
                            ''.format(name))

        tc_in, _ = sig.split('->')
        a = array(a, dtype=tc_in, copy=False, order=order, context=context,
                  cls=cls)
    else:
        a = array(a, context=context, cls=cls)

        sig = find_smallest_valid_signature(name, (a,), (out,))
        if not sig:
            raise TypeError('ufunc {!r} not supported for the input types, '
                            'and the inputs could not be safely coerced'
                            ''.format(name))

        # Upcast input if necessary
        tc_in, tc_out = sig.split('->')
        if a.dtype < tc_in:
            a = a.astype(tc_in)

    # Create output array if not provided
    if out is None:
        out = empty(a.shape, dtype=tc_out, context=context, cls=cls)

    # --- Generate code strings for GpuElemwise --- #

    # C dtypes for casting
    c_dtype_in = dtype_to_ctype(tc_in)
    c_dtype_out = dtype_to_ctype(tc_out)

    meta = ufunc_metadata[name]
    assert meta['nin'] == 1
    assert meta['nout'] == 1

    # Create `oper` string
    if meta['c_op'] is not None:
        # Case 1: unary operator
        unop = meta['c_op']
        if a.dtype == numpy.bool and unop == '-':
            if parse_version(numpy.__version__) >= parse_version('1.13'):
                # Numpy >= 1.13 raises a TypeError
                raise TypeError(
                    'negation of boolean arrays is not supported, use '
                    '`logical_not` instead')
            else:
                # Warn and remap to logical not
                warnings.warn('using negation (`-`) with boolean arrays is '
                              'deprecated, use `logical_not` (`~`) instead; '
                              'the current behavior will be changed along '
                              "with NumPy's", FutureWarning)
                unop = '!'
        oper = 'out = ({odt}) {}a'.format(unop, odt=c_dtype_out)
        preamble = ''

    elif meta['c_func'] is not None:
        # Case 2: C function
        c_func = meta['c_func']

        if name in ('abs', 'absolute'):
            # Special case
            if numpy.dtype(tc_out).kind == 'u':
                # Shortcut for abs() with unsigned int. This also fixes a CUDA
                # quirk that makes abs() crash with unsigned int input.
                out[:] = a
                return out
            elif numpy.dtype(tc_out).kind == 'f':
                c_func = 'fabs'
            else:
                c_func = 'abs'

        oper = 'out = ({odt}) {}(a)'.format(c_func, odt=c_dtype_out)
        preamble_tpl = mako.template.Template(meta['oper_preamble_tpl'])
        preamble = preamble_tpl.render(idt=c_dtype_in, odt=c_dtype_out)

    elif meta['oper_fmt'] is not None:
        # Case 3: custom implementation with `oper` template
        oper = meta['oper_fmt'].format(idt=c_dtype_in, odt=c_dtype_out)
        preamble_tpl = mako.template.Template(meta['oper_preamble_tpl'])
        preamble = preamble_tpl.render(idt=c_dtype_in, odt=c_dtype_out)

    else:
        # Case 4: not implemented
        raise NotImplementedError('ufunc {!r} not implemented'.format(name))

    # --- Generate and run GpuElemwise kernel --- #

    a_arg = as_argument(a, 'a', read=True)
    args = [arg('out', out.dtype, write=True), a_arg]

    ker = GpuElemwise(context, oper, args, preamble=preamble)
    ker(out, a)
    return out


# %% Test

import pygpu
ctx = pygpu.init('cuda')
pygpu.set_default_context(ctx)
