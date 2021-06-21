.. _api_helpers:

****************
Helper Functions
****************

These are helper functions that control aspects of the hipSOLVER library. These are divided
into two categories:

* :ref:`initialize` functions. Used to initialize and cleanup the library handle.
* :ref:`stream` functions. Provide functionality to manipulate streams.


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

