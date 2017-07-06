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

        # Defined by us

        # Docstring
        entry['doc'] = ''
        # Alternative names, for duplication in module namespace
        entry['alt_names'] = ()
        # Valid range of inputs written as, e.g., '[-1, 1]' or
        # '[-inf, -1) + (1, inf]'. Only the '+' symbol can be used to
        # as "union".
        # This is mostly for testing purposes (to determine valid test input).
        entry['domain'] = ''
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
        # Here, `c_cast_dtype_*` are strings of C dtypes used for casting
        # of input and output data types, e.g., for `true_divide`:
        #
        #     oper_fmt = 'res = ({odt}) (({idt1}) a / ({idt2}) b)'
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
        # including unsupported data types. Each entry is itself a
        # dictionary with the following possible items:
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
        # Present keys (except 'reason') are AND'ed together.
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

# --- add --- #

ufunc_metadata['add']['c_op'] = '+'

# doc
# alt_names
# domain
# c_func
# c_op
# oper_fmt
# oper_preamble_tpl
# npy_incompat

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

ufunc_metadata['cbrt']['c_func'] = 'cbrt'

# --- ceil --- #

ufunc_metadata['ceil']['c_func'] = 'ceil'

# --- conjugate --- #

# Leave unimplemented by not adding anything

# --- copysign --- #
# --- cos --- #
# --- cosh --- #
# --- deg2rad --- #
# --- degrees --- #
# --- equal --- #
# --- exp --- #
# --- exp2 --- #
# --- expm1 --- #
# --- fabs --- #
# --- float_power --- #
# --- floor --- #
# --- floor_divide --- #
# --- fmax --- #
# --- fmin --- #
# --- fmod --- #
# --- frexp --- #
# --- greater --- #
# --- greater_equal --- #
# --- hypot --- #
# --- invert --- #
# --- isfinite --- #
# --- isinf --- #
# --- isnan --- #
# --- ldexp --- #
# --- left_shift --- #
# --- less --- #
# --- less_equal --- #
# --- log --- #
# --- log10 --- #
# --- log1p --- #
# --- log2 --- #
# --- logaddexp --- #
# --- logaddexp2 --- #
# --- logical_and --- #
# --- logical_not --- #
# --- logical_or --- #
# --- logical_xor --- #
# --- maximum --- #
# --- minimum --- #
# --- modf --- #
# --- multiply --- #
# --- negative --- #
# --- nextafter --- #
# --- not_equal --- #
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
