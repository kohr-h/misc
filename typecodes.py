#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:25:39 2017

@author: kohr
"""
import numpy

# Not supported:
# - 'e' (float16)
# - 'FDG' (complex64/128/256)
# - 'SU' (str/unicode)
# - 'V' (void)
# - 'O' (object)
# - 'Mm' (datetime)
SUPP_TYPECODES = '?bhilqpBHILQPfdg'
SUPP_TYPECODES_TO_DTYPES = {tc: numpy.dtype(tc) for tc in SUPP_TYPECODES}


def metadata_from_numpy():
    """Initialize the metadata dictionary from Numpy."""
    meta = {}
    ufuncs = []
    for name in dir(numpy):
        obj = getattr(numpy, name)
        if isinstance(obj, numpy.ufunc) and name == obj.__name__:
            ufuncs.append(obj)

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

        # Valid range of inputs written as, e.g., '[-1, 1]' or
        # '[-inf, -1) + (1, inf]'. Only the '+' symbol can be used to
        # as "union".
        # This is mostly for testing purposes (to determine valid test input).
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
        # `oper_fmt.format()`.
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

        meta[ufunc.__name__] = entry

    return meta


ufunc_metadata = metadata_from_numpy()


def patch_types(types):
    """Return a new list with unsupported type signatures removed."""
    new_types = []
    for sig in types:
        ops, res = sig.split('->')
        if not (all(c in SUPP_TYPECODES for c in ops) and
                all(c in SUPP_TYPECODES for c in res)):
            # Signature contains unsupported type, not adding
            continue
        else:
            new_types.append(sig)
    return new_types


for meta in ufunc_metadata.values():
    meta['types'] = patch_types(meta['types'])
    meta['ntypes'] = len(meta['types'])


# %% Setting the individual metadata

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

ufunc_metadata['deg2rad']['npy_ver_added'] = '1.3.0'
_oper_fmt = 'out = ({{odt}})({:.45f} * ({{idt}}) a)'.format(numpy.deg2rad(1.0))
ufunc_metadata['deg2rad']['oper_fmt'] = _oper_fmt

# --- degrees --- #

_oper_fmt = 'out = ({{odt}})({:.45f} * ({{idt}}) a)'.format(numpy.degrees(1.0))
ufunc_metadata['degrees']['oper_fmt'] = _oper_fmt

# --- divmod --- #

_preamble_tpl = '''
WITHIN_KERNEL ${odt1}
divmod(${idt1} a, ${idt2} b, ${odt2} *out2) {
    ${odt1} rem = fmod(a, b);
    *out2 = a / b;
    return rem;
}
'''
_oper_fmt = 'out1 = ({odt1}) divmod(a, b, &out2)'
ufunc_metadata['divmod']['oper_fmt'] = _oper_fmt
ufunc_metadata['divmod']['oper_preamble_tpl'] = _preamble_tpl

# --- equal --- #

ufunc_metadata['equal']['c_op'] = '=='

# --- exp --- #

ufunc_metadata['exp']['c_func'] = 'exp'

# --- exp2 --- #

ufunc_metadata['exp2']['c_func'] = 'exp2'
ufunc_metadata['exp2']['npy_ver_added'] = '1.3.0'

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
    ${odt} quot = a / b;
    if ((a < 0) != (b < 0)) {
        return (${odt}) -(quot + 0.999);
    } else {
        return (${odt}) quot;
    }
}
'''
_oper_fmt = 'out = ({odt}) floor_divide(a, b)'
ufunc_metadata['floor_divide']['oper_fmt'] = _oper_fmt
ufunc_metadata['floor_divide']['oper_preamble_tpl'] = _preamble_tpl

# --- fmax --- #

# TODO: same as maximum, except for NaN propagation
ufunc_metadata['fmax']['npy_ver_added'] = '1.3.0'

# --- fmin --- #

# TODO: same as minimum, except for NaN propagation
ufunc_metadata['fmax']['npy_ver_added'] = '1.3.0'

# --- fmod --- #

ufunc_metadata['fmod']['c_func'] = 'fmod'

# --- frexp --- #

_oper_fmt = 'out = ({odt1}) frexp(({idt}) a, &out2)'
ufunc_metadata['frexp']['oper_fmt'] = _oper_fmt

# --- greater --- #

ufunc_metadata['greater']['c_op'] = '>'

# --- greater_equal --- #

ufunc_metadata['greater_equal']['c_op'] = '>='

# --- heaviside --- #

_preamble_tpl = '''
WITHIN_KERNEL ${odt}
heaviside(${idt1} a, ${idt2} b) {
    if (a < 0) {
        return (${odt}) 0;
    }
    else if (a == 0) {
        return (${odt}) b;
    }
    else {
        return (${odt}) 1;
    }
}
'''
_oper_fmt = 'out = ({odt}) heaviside(a, b)'
ufunc_metadata['heaviside']['oper_fmt'] = _oper_fmt
ufunc_metadata['heaviside']['oper_preamble_tpl'] = _preamble_tpl
ufunc_metadata['heaviside']['npy_ver_added'] = '1.13.0'

# --- hypot --- #

ufunc_metadata['hypot']['c_func'] = 'hypot'

# --- invert --- #

ufunc_metadata['invert']['c_op'] = '~'

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

# --- log10 --- #

ufunc_metadata['log10']['c_func'] = 'log10'

# --- log1p --- #

ufunc_metadata['log1p']['c_func'] = 'log1p'

# --- log2 --- #

ufunc_metadata['log2']['c_func'] = 'log2'

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
# --- minimum --- #
# --- modf --- #
# --- multiply --- #
# --- negative --- #
# --- nextafter --- #
# --- not_equal --- #
# --- positive --- #

# new in Numpy 1.13

# --- power --- #
# --- rad2deg --- #
# --- radians --- #
# --- reciprocal --- #
# --- remainder --- #
# --- right_shift --- #
# --- rint --- #
# --- sign --- #
# --- signbit --- #
# --- sin --- #
# --- sinh --- #
# --- spacing --- #
# --- sqrt --- #
# --- square --- #
# --- subtract --- #
# --- tan --- #
# --- tanh --- #
# --- true_divide --- #
# --- trunc --- #

_oper_fmt = None
