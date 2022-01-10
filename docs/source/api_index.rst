.. _library_api:

########################################
hipSOLVER API
########################################

Currently, this API document only provides the method signatures for the wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backends' documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/index.html#cuds-api>`_ and/or `rocSOLVER API <https://rocsolver.readthedocs.io/en/latest/api_index.html>`_.

The hipSOLVER API is designed to be similar to the cusolverDn and rocSOLVER interfaces, but it requires some minor adjustments to ensure the best performance out of 
either backend. Generally, this involves the addition of workspace parameters and some additional API methods. 
Please refer to the user guide for a complete listing of :ref:`these API adjustments <api_differences>`.

Users interested in using hipSOLVER without these adjustments, having an interface that matches cuSOLVER, should instead consult the :ref:`compatibility API documentation <library_compat>`.
See also :ref:`this section <porting>` for more details.

.. toctree::
   :maxdepth: 5

   api_types
   api_helperfunc
   api_auxiliaryfunc
   api_lapackfunc

