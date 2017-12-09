travis-sphinx build -n -s doc/source/ 2>sphinx_errors.log
cat sphinx_errors.log | grep -E -v "WARNING: toctree references unknown document 'generated/" \
                      | grep -E -v "WARNING: toctree contains reference to nonexisting document 'generated/" \
                      | grep -E -v "WARNING: more than one target found for 'any' cross-reference 'Operator': could be :std:term: or :py:class:" \
                      | grep -E -v "WARNING: py:obj reference target not found: odl." \
