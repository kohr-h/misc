# Integration of C code in a Python project


## 1. Write new code

### Variant A: Distribute with the package
* Write a `.pyx` file in a reasonable location (where the package should end up in the tree), e.g., `package/module/extension.pyx`
* Add the relative path to the cython_modules list in `setup.py`
* The code in `setup.py` will automatically create a shared library from the `.pyx` file in the same location, e.g., `package/module/extension.cpython-36m-x86_64-linux-gnu.so`.
* This shared library can be imported in a Python script, e.g., `from . import extension` in the same directory.

### Variant B: Make a separate extension
* Create a minimal `setup.py` with the Cython-related code, plus some information about the extension. Code can probably be copied from the main `setup.py` to save some time.
* Create the `.pyx` file in the top level if only one file is desired. For more than one, create a package structure with a package directory and add `.pyx` files as submodules.
* Import the external package and catch `ImportError`. Set `<PKG>_AVAILABLE` to `False` if the package is not present (i.e., not compiled). Fence all usages of the package with queries to that flag.
* Add the extension to the optional dependencies, perhaps as a Git submodule or some other way to pull in compiled extensions (conda package, wheel, ...).


## 2. Wrap existing code
* For each `foo.h` file, create a `foo.pxd` file and replicate the API using the Cython language. This can be restricted to the calls that are needed in Python.
* For each `foo.c` file, create a `foo_c.pyx` file that wraps the underlying C implementation. The base name of the `.pyx` file must be different from the one of the `.c` file since Cython will generate an intermediate C file that would overwrite the original C file.
* Use the same `setup.py` code as in 1. to generate a Cython extension. As `'sources'`, you need `foo.c` and `foo.h` (and others that are involved), as well as (potentially) the `.pxd` files. If NumPy is involved, its headers must be added to the `'include_dirs'`.


## Resources
- https://github.com/kohr-h/variable_lp_paper -- package using Cython extensions
- https://github.com/kohr-h/variable_lp_paper/blob/master/setup.py -- example `setup.py` using Cython
