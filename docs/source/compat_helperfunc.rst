.. _compat_helpers:

****************
Helper Functions
****************

These are helper functions that control aspects of the hipSOLVER library. These are divided
into two categories:

* :ref:`compat_initialize` functions. Used to initialize and cleanup the library handle.
* :ref:`compat_stream` functions. Provide functionality to manipulate streams.
* :ref:`compat_gesvdj_info` functions. Provide functionality to manipulate gesvdj parameters.


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

