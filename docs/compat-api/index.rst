.. _library_compat:

##############################################
hipSOLVER Compatibility API (Dense Matrices)
##############################################

Currently, this API document only provides the method signatures for the wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backends' documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/>`_ and/or `rocSOLVER API <https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/api/index.html>`_.

For ease of porting from existing cuSOLVER applications to hipSOLVER, functions in the hipsolverDn compatibility API are designed to have
method signatures that are consistent with the cusolverDn interface. However, :ref:`performance issues <compat_performance>` may arise when
using the rocSOLVER backend due to differing workspace requirements. Therefore, users interested in achieving the best performance with
the rocSOLVER backend should consult the :ref:`regular API documentation <library_api>`, and transition from the compatibility API to
the regular API at the earliest convenience. Please refer to the user guide for additional :ref:`considerations regarding the use of
the compatibility API <compat_api_differences>`.

.. tableofcontents::
