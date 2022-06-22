.. _api_helpers:

****************
Helper Functions
****************

These are helper functions that control aspects of the hipSOLVER library. These are divided
into two categories:

* :ref:`initialize` functions. Used to initialize and cleanup the library handle.
* :ref:`stream` functions. Provide functionality to manipulate streams.
* :ref:`syevj_info` functions. Provide functionality to manipulate syevj parameters.


.. _initialize:

Handle set-up and tear-down
===============================

.. contents:: List of handle initialization functions
   :local:
   :backlinks: top

hipsolverCreate()
---------------------------------
.. doxygenfunction:: hipsolverCreate

hipsolverDestroy()
---------------------------------
.. doxygenfunction:: hipsolverDestroy



.. _stream:

Stream manipulation
==============================

.. contents:: List of stream manipulation functions
   :local:
   :backlinks: top

hipsolverSetStream()
---------------------------------
.. doxygenfunction:: hipsolverSetStream

hipsolverGetStream()
---------------------------------
.. doxygenfunction:: hipsolverGetStream



.. _syevj_info:

Syevj parameter manipulation
===============================

.. contents:: List of syevj parameter functions
   :local:
   :backlinks: top

hipsolverCreateSyevjInfo()
---------------------------------
.. doxygenfunction:: hipsolverCreateSyevjInfo

hipsolverDestroySyevjInfo()
---------------------------------
.. doxygenfunction:: hipsolverDestroySyevjInfo

.. _syevj_set_max_sweeps:

hipsolverXsyevjSetMaxSweeps()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjSetMaxSweeps

.. _syevj_set_sort_eig:

hipsolverXsyevjSetSortEig()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjSetSortEig

.. _syevj_set_tolerance:

hipsolverXsyevjSetTolerance()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjSetTolerance

.. _syevj_get_residual:

hipsolverXsyevjGetResidual()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjGetResidual

.. _syevj_get_sweeps:

hipsolverXsyevjGetSweeps()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjGetSweeps

