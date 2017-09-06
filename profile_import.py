from import_profiler import profile_import

with profile_import() as prof:
    import odl

prof.print_info()
