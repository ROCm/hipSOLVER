
.. _lapacklike:

**********************
LAPACK-like Functions
**********************

Other Lapack-like routines provided by hipSOLVER. These are divided into the following subcategories:

* :ref:`likeeigens`. Eigenproblems for symmetric matrices.



.. _likeeigens:

Symmetric eigensolvers
================================

.. contents:: List of Lapack-like symmetric eigensolvers
   :local:
   :backlinks: top

.. _syevj_bufferSize:

hipsolver<type>syevj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsyevj_bufferSize

.. _heevj_bufferSize:

hipsolver<type>heevj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCheevj_bufferSize

.. _syevj_batched_bufferSize:

hipsolver<type>syevjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsyevjBatched_bufferSize

.. _heevj_batched_bufferSize:

hipsolver<type>heevjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCheevjBatched_bufferSize

.. _syevj:

hipsolver<type>syevj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevj
   :outline:
.. doxygenfunction:: hipsolverSsyevj

.. _heevj:

hipsolver<type>heevj()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevj
   :outline:
.. doxygenfunction:: hipsolverCheevj

.. _syevj_batched:

hipsolver<type>syevjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevjBatched
   :outline:
.. doxygenfunction:: hipsolverSsyevjBatched

.. _heevj_batched:

hipsolver<type>heevjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevjBatched
   :outline:
.. doxygenfunction:: hipsolverCheevjBatched

.. _sygvj_bufferSize:

hipsolver<type>sygvj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsygvj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsygvj_bufferSize

.. _hegvj_bufferSize:

hipsolver<type>hegvj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhegvj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverChegvj_bufferSize

.. _sygvj:

hipsolver<type>sygvj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsygvj
   :outline:
.. doxygenfunction:: hipsolverSsygvj

.. _hegvj:

hipsolver<type>hegvj()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhegvj
   :outline:
.. doxygenfunction:: hipsolverChegvj

