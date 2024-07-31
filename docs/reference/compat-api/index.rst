.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _library_compat:

********************************************************************
hipSOLVER compatibility API - Dense Matrices
********************************************************************

This document provides the method signatures for the wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backend documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/>`_ and/or :doc:`rocSOLVER API <rocsolver:reference/intro>`.

For ease of porting from existing cuSOLVER applications to hipSOLVER, functions in the hipsolverDn compatibility API are designed to have
method signatures that are consistent with the cusolverDn interface. However, :ref:`performance issues <compat_performance>` may arise when
using the rocSOLVER backend due to differing workspace requirements. Therefore, users interested in achieving the best performance with
the rocSOLVER backend should consult the :ref:`regular API documentation <library_api>`, and transition from the compatibility API to
the regular API at the earliest convenience. Please refer to :ref:`usage_label` for additional :ref:`considerations regarding the use of
the compatibility API <compat_api_differences>`.

  * :ref:`compat_types`
  * :ref:`compat_helpers` 
  * :ref:`compat_auxiliary`
  * :ref:`compat_lapackfunc`
  * :ref:`compat_lapacklike`

