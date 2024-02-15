.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _library_refactor:

********************************************************************
hipSOLVER Compatibility API - Refactorization
********************************************************************

This document provides the method signatures for the wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backend documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/index.html#cuds-api>`_ and/or `rocSOLVER API <https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/api/index.html>`_.

For ease of porting from existing cuSOLVER applications to hipSOLVER, functions in the hipsolverRf compatibility API are designed to have
method signatures that are consistent with the cusolverRf interface. At present, equivalent functions have not been added to hipSOLVER's regular API.

  * :ref:`refactor_types`
  * :ref:`refactor_helpers` 
  * :ref:`refactor_refactorfunc`
