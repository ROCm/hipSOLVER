.. _compat_helpers:

****************
Helper Functions
****************

These are helper functions that control aspects of the hipSOLVER library. These are divided
into two categories:

* :ref:`compat_initialize` functions. Used to initialize and cleanup the library handle.
* :ref:`compat_stream` functions. Provide functionality to manipulate streams.


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

