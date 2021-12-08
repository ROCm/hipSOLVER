.. _library_compat:

########################################
hipSOLVER Compatibility API
########################################

Currently, this API document only provides the method signatures for the wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backends' documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/index.html#cuds-api>`_ and/or `rocSOLVER API <https://rocsolver.readthedocs.io/en/latest/api_index.html>`_.

For ease of porting, functions in hipSOLVER's compatibility API are designed to have method signatures that are consistent with the
cusolverDn interface. However, performance issues may arise when using the rocSOLVER backend due to differing workspace requirements.
Therefore, transitioning from the compatibility API to the regular API is recommended at the earliest convenience. Please refer to the
user guide for additional considerations regarding :ref:`arguments that are not referenced <unused_arguments>` by the rocSOLVER backend.

Users interested in achieving the best performance with the rocSOLVER backend should consult the :ref:`regular API documentation <library_api>`.

.. toctree::
   :maxdepth: 5

   compat_types
   compat_helperfunc
   compat_auxiliaryfunc
   compat_lapackfunc

