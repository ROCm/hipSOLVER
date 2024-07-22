.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _dense_helpers:

*****************************************
Dense matrix helper functions
*****************************************

These are helper functions that control aspects of the hipSOLVER library. They are divided
into the following categories:

* :ref:`dense_initialize` functions used to initialize and cleanup the library handle.
* :ref:`dense_stream` functions provide functionality to manipulate streams.
* :ref:`dense_determinism` functions. Provide functionality to manipulate function determinism.
* :ref:`dense_gesvdj_info` functions provide functionality to manipulate gesvdj parameters.
* :ref:`dense_syevj_info` functions provide functionality to manipulate syevj parameters.
* :ref:`dense_params` functions provide functionality to manipulate other parameters.


.. _dense_initialize:

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



.. _dense_stream:

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



.. _dense_determinism:

Determinism manipulation
==============================

.. contents:: List of deterministic mode manipulation functions
   :local:
   :backlinks: top

hipsolverDnSetDeterministicMode()
----------------------------------
.. doxygenfunction:: hipsolverDnSetDeterministicMode

hipsolverDnGetDeterministicMode()
----------------------------------
.. doxygenfunction:: hipsolverDnGetDeterministicMode



.. _dense_gesvdj_info:

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

.. _dense_gesvdj_set_max_sweeps:

hipsolverDnXgesvdjSetMaxSweeps()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjSetMaxSweeps

.. _dense_gesvdj_set_sort_eig:

hipsolverDnXgesvdjSetSortEig()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjSetSortEig

.. _dense_gesvdj_set_tolerance:

hipsolverDnXgesvdjSetTolerance()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjSetTolerance

.. _dense_gesvdj_get_residual:

hipsolverDnXgesvdjGetResidual()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjGetResidual

.. _dense_gesvdj_get_sweeps:

hipsolverDnXgesvdjGetSweeps()
---------------------------------
.. doxygenfunction:: hipsolverDnXgesvdjGetSweeps



.. _dense_syevj_info:

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

.. _dense_syevj_set_max_sweeps:

hipsolverDnXsyevjSetMaxSweeps()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjSetMaxSweeps

.. _dense_syevj_set_sort_eig:

hipsolverDnXsyevjSetSortEig()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjSetSortEig

.. _dense_syevj_set_tolerance:

hipsolverDnXsyevjSetTolerance()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjSetTolerance

.. _dense_syevj_get_residual:

hipsolverDnXsyevjGetResidual()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjGetResidual

.. _dense_syevj_get_sweeps:

hipsolverDnXsyevjGetSweeps()
---------------------------------
.. doxygenfunction:: hipsolverDnXsyevjGetSweeps



.. _dense_params:

Other parameter manipulation
===============================

.. contents:: List of other parameter functions
   :local:
   :backlinks: top

hipsolverDnCreateParams()
---------------------------------
.. doxygenfunction:: hipsolverDnCreateParams

hipsolverDnDestroyParams()
---------------------------------
.. doxygenfunction:: hipsolverDnDestroyParams

hipsolverDnSetAdvOptions()
---------------------------------
.. doxygenfunction:: hipsolverDnSetAdvOptions

