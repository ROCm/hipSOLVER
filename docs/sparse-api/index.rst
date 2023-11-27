.. _library_sparse:

##############################################
hipSOLVER Compatibility API (Sparse Matrices)
##############################################

Currently, this API document only provides the method signatures for the wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backends' documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/index.html#cuds-api>`_ and/or `rocSOLVER API <https://rocsolver.readthedocs.io/en/latest/api_index.html>`_.

For ease of porting from existing cuSOLVER applications to hipSOLVER, functions in the hipsolverSp compatibility API are designed to have
method signatures that are consistent with the cusolverSp interface. At present, equivalent functions have not been added to hipSOLVER's
regular API. Note that there are :ref:`some performance limitations <sparse_performance>` when using the rocSOLVER backend as not all the
functionality required for optimal performance has been implemented yet.

.. tableofcontents::

