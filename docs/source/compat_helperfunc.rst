.. _compat_helpers:

****************
Helper Functions
****************

These are helper functions that control aspects of the hipSOLVER library. These are divided
into two categories:

* :ref:`compat_initialize` functions. Used to initialize and cleanup the library handle.
* :ref:`compat_stream` functions. Provide functionality to manipulate streams.
* :ref:`compat_syevj_info` functions. Provide functionality to manipulate syevj parameters.


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

