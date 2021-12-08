.. _library_api:

########################################
hipSOLVER API
########################################

Currently, this API document only provides the method signatures for the wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backends' documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/index.html#cuds-api>`_ and/or `rocSOLVER API <https://rocsolver.readthedocs.io/en/latest/api_index.html>`_.

The hipSOLVER API is designed to be similar to the cusolverDn interface, but includes some minor adjustments for the sake of the rocSOLVER
backend. Generally, this involves the addition of workspace parameters `work` and `lwork` to those functions that do not otherwise require
both workspace arguments. Please refer to the user guide for a complete listing of :ref:`these parameter adjustments <api_differences>`,
along with a listing of :ref:`those arguments that are not referenced <unused_arguments>` by the rocSOLVER backend.

Users interested in using hipSOLVER without these adjustments should instead consult the :ref:`compatibility API documentation <library_compat>`.

.. toctree::
   :maxdepth: 5

   api_types
   api_helperfunc
   api_auxiliaryfunc
   api_lapackfunc

