.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _compat_helpers:

*****************************************
Dense matrix helper functions
*****************************************

These are helper functions that control aspects of the hipSOLVER library. They are divided
into the following categories:

* :ref:`compat_initialize` functions used to initialize and cleanup the library handle
* :ref:`compat_stream` functions provide functionality to manipulate streams
* :ref:`compat_gesvdj_info` functions provide functionality to manipulate gesvdj parameters
* :ref:`compat_syevj_info` functions provide functionality to manipulate syevj parameters


.. _compat_initialize:

Handle set-up and tear-down
===============================

.. contents:: List of handle initialization functions
   :local:
   :backlinks: top

hipsolverDnCreate()
---------------------------------
.. doxygenfunction:: hipsolverDnCreate

hipsolverDnDestroy()
---------------------------------
.. doxygenfunction:: hipsolverDnDestroy



.. _compat_stream:

Stream manipulation
==============================

.. contents:: List of stream manipulation functions
   :local:
   :backlinks: top

hipsolverDnSetStream()
---------------------------------
.. doxygenfunction:: hipsolverDnSetStream

hipsolverDnGetStream()
---------------------------------
.. doxygenfunction:: hipsolverDnGetStream



.. _compat_gesvdj_info:

Gesvdj parameter manipulation
===============================

.. contents:: List of gesvdj parameter functions
   :local:
   :backlinks: top

hipsolverDnCreateGesvdjInfo()
---------------------------------
.. doxygenfunction:: hipsolverDnCreateGesvdjInfo

hipsolverDnDestroyGesvdjInfo()
---------------------------------
.. doxygenfunction:: hipsolverDnDestroyGesvdjInfo

.. _compat_gesvdj_set_max_sweeps:

hipsolverDnXgesvdjSetMaxSweeps()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjSetMaxSweeps

.. _compat_gesvdj_set_sort_eig:

hipsolverDnXgesvdjSetSortEig()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjSetSortEig

.. _compat_gesvdj_set_tolerance:

hipsolverDnXgesvdjSetTolerance()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjSetTolerance

.. _compat_gesvdj_get_residual:

hipsolverDnXgesvdjGetResidual()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjGetResidual

.. _compat_gesvdj_get_sweeps:

hipsolverDnXgesvdjGetSweeps()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjGetSweeps



.. _compat_syevj_info:

Syevj parameter manipulation
===============================

.. contents:: List of syevj parameter functions
   :local:
   :backlinks: top

hipsolverDnCreateSyevjInfo()
---------------------------------
.. doxygenfunction:: hipsolverDnCreateSyevjInfo

hipsolverDnDestroySyevjInfo()
---------------------------------
.. doxygenfunction:: hipsolverDnDestroySyevjInfo

.. _compat_syevj_set_max_sweeps:

hipsolverDnXsyevjSetMaxSweeps()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjSetMaxSweeps

.. _compat_syevj_set_sort_eig:

hipsolverDnXsyevjSetSortEig()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjSetSortEig

.. _compat_syevj_set_tolerance:

hipsolverDnXsyevjSetTolerance()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjSetTolerance

.. _compat_syevj_get_residual:

hipsolverDnXsyevjGetResidual()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjGetResidual

.. _compat_syevj_get_sweeps:

hipsolverDnXsyevjGetSweeps()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjGetSweeps

