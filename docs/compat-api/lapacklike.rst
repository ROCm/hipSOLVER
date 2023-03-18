
.. _compat_lapacklike:

**********************
LAPACK-like Functions
**********************

Other Lapack-like routines provided by hipSOLVER. These are divided into the following subcategories:

* :ref:`compat_likeeigens`. Eigenproblems for symmetric matrices.
* :ref:`compat_likesvds`. Singular values and related problems for general matrices.



.. _compat_likeeigens:

Symmetric eigensolvers
================================

.. contents:: List of Lapack-like symmetric eigensolvers
   :local:
   :backlinks: top

.. _compat_syevdx_bufferSize:

hipsolverDn<type>syevdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsyevdx_bufferSize

.. _compat_heevdx_bufferSize:

hipsolverDn<type>heevdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCheevdx_bufferSize

.. _compat_syevdx:

hipsolverDn<type>syevdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevdx
   :outline:
.. doxygenfunction:: hipsolverDnSsyevdx

.. _compat_heevdx:

hipsolverDn<type>heevdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevdx
   :outline:
.. doxygenfunction:: hipsolverDnCheevdx

.. _compat_syevj_bufferSize:

hipsolverDn<type>syevj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsyevj_bufferSize

.. _compat_heevj_bufferSize:

hipsolverDn<type>heevj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCheevj_bufferSize

.. _compat_syevj_batched_bufferSize:

hipsolverDn<type>syevjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsyevjBatched_bufferSize

.. _compat_heevj_batched_bufferSize:

hipsolverDn<type>heevjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCheevjBatched_bufferSize

.. _compat_syevj:

hipsolverDn<type>syevj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevj
   :outline:
.. doxygenfunction:: hipsolverDnSsyevj

.. _compat_heevj:

hipsolverDn<type>heevj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevj
   :outline:
.. doxygenfunction:: hipsolverDnCheevj

.. _compat_syevj_batched:

hipsolverDn<type>syevjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevjBatched
   :outline:
.. doxygenfunction:: hipsolverDnSsyevjBatched

.. _compat_heevj_batched:

hipsolverDn<type>heevjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevjBatched
   :outline:
.. doxygenfunction:: hipsolverDnCheevjBatched

.. _compat_sygvdx_bufferSize:

hipsolverDn<type>sygvdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsygvdx_bufferSize

.. _compat_hegvdx_bufferSize:

hipsolverDn<type>hegvdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnChegvdx_bufferSize

.. _compat_sygvdx:

hipsolverDn<type>sygvdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvdx
   :outline:
.. doxygenfunction:: hipsolverDnSsygvdx

.. _compat_hegvdx:

hipsolverDn<type>hegvdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvdx
   :outline:
.. doxygenfunction:: hipsolverDnChegvdx

.. _compat_sygvj_bufferSize:

hipsolverDn<type>sygvj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsygvj_bufferSize

.. _compat_hegvj_bufferSize:

hipsolverDn<type>hegvj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnChegvj_bufferSize

.. _compat_sygvj:

hipsolverDn<type>sygvj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvj
   :outline:
.. doxygenfunction:: hipsolverDnSsygvj

.. _compat_hegvj:

hipsolverDn<type>hegvj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvj
   :outline:
.. doxygenfunction:: hipsolverDnChegvj



.. _compat_likesvds:

Singular value decomposition
================================

.. contents:: List of Lapack-like SVD related functions
   :local:
   :backlinks: top

.. _compat_gesvdj_bufferSize:

hipsolverDn<type>gesvdj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdj_bufferSize

.. _compat_gesvdj_batched_bufferSize:

hipsolverDn<type>gesvdjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdjBatched_bufferSize

.. _compat_gesvdj:

hipsolverDn<type>gesvdj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdj
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdj
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdj
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdj

.. _compat_gesvdj_batched:

hipsolverDn<type>gesvdjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdjBatched
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdjBatched
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdjBatched
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdjBatched

.. _compat_gesvda_strided_batched_bufferSize:

hipsolverDn<type>gesvdaStridedBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdaStridedBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdaStridedBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdaStridedBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdaStridedBatched_bufferSize

.. _compat_gesvda_strided_batched:

hipsolverDn<type>gesvdaStridedBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdaStridedBatched
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdaStridedBatched
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdaStridedBatched
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdaStridedBatched

