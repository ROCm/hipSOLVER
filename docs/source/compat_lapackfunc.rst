
.. _compat_lapackfunc:

********************
LAPACK Functions
********************

LAPACK routines solve complex Numerical Linear Algebra problems. These functions are organized
in the following categories:

* :ref:`compat_triangular`. Based on Gaussian elimination.
* :ref:`compat_orthogonal`. Based on Householder reflections.
* :ref:`compat_reductions`. Transformation of matrices and problems into equivalent forms.
* :ref:`compat_linears`. Based on triangular factorizations.
* :ref:`compat_leastsqr`. Based on orthogonal factorizations.
* :ref:`compat_eigens`. Eigenproblems for symmetric matrices.
* :ref:`compat_svds`. Singular values and related problems for general matrices.



.. _compat_triangular:

Triangular factorizations
================================

.. contents:: List of triangular factorizations
   :local:
   :backlinks: top

.. _compat_potrf_bufferSize:

hipsolverDn<type>potrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSpotrf_bufferSize

.. _compat_potrf:

hipsolverDn<type>potrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotrf
   :outline:
.. doxygenfunction:: hipsolverDnCpotrf
   :outline:
.. doxygenfunction:: hipsolverDnDpotrf
   :outline:
.. doxygenfunction:: hipsolverDnSpotrf

.. _compat_potrf_batched:

hipsolverDn<type>potrfBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotrfBatched
   :outline:
.. doxygenfunction:: hipsolverDnCpotrfBatched
   :outline:
.. doxygenfunction:: hipsolverDnDpotrfBatched
   :outline:
.. doxygenfunction:: hipsolverDnSpotrfBatched

.. _compat_getrf_bufferSize:

hipsolverDn<type>getrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgetrf_bufferSize

.. _compat_getrf:

hipsolverDn<type>getrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgetrf
   :outline:
.. doxygenfunction:: hipsolverDnCgetrf
   :outline:
.. doxygenfunction:: hipsolverDnDgetrf
   :outline:
.. doxygenfunction:: hipsolverDnSgetrf

.. _compat_sytrf_bufferSize:

hipsolverDn<type>sytrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZsytrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCsytrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDsytrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsytrf_bufferSize

.. _compat_sytrf:

hipsolverDn<type>sytrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZsytrf
   :outline:
.. doxygenfunction:: hipsolverDnCsytrf
   :outline:
.. doxygenfunction:: hipsolverDnDsytrf
   :outline:
.. doxygenfunction:: hipsolverDnSsytrf



.. _compat_orthogonal:

Orthogonal factorizations
================================

.. contents:: List of orthogonal factorizations
   :local:
   :backlinks: top

.. _compat_geqrf_bufferSize:

hipsolverDn<type>geqrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgeqrf_bufferSize

.. _compat_geqrf:

hipsolverDn<type>geqrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgeqrf
   :outline:
.. doxygenfunction:: hipsolverDnCgeqrf
   :outline:
.. doxygenfunction:: hipsolverDnDgeqrf
   :outline:
.. doxygenfunction:: hipsolverDnSgeqrf



.. _compat_reductions:

Problem and matrix reductions
================================

.. contents:: List of reductions
   :local:
   :backlinks: top

.. _compat_gebrd_bufferSize:

hipsolverDn<type>gebrd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgebrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgebrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgebrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgebrd_bufferSize

.. _compat_gebrd:

hipsolverDn<type>gebrd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgebrd
   :outline:
.. doxygenfunction:: hipsolverDnCgebrd
   :outline:
.. doxygenfunction:: hipsolverDnDgebrd
   :outline:
.. doxygenfunction:: hipsolverDnSgebrd

.. _compat_sytrd_bufferSize:

hipsolverDn<type>sytrd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsytrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsytrd_bufferSize

.. _compat_hetrd_bufferSize:

hipsolverDn<type>hetrd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhetrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnChetrd_bufferSize

.. _compat_sytrd:

hipsolverDn<type>sytrd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsytrd
   :outline:
.. doxygenfunction:: hipsolverDnSsytrd

.. _compat_hetrd:

hipsolverDn<type>hetrd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhetrd
   :outline:
.. doxygenfunction:: hipsolverDnChetrd



.. _compat_linears:

Linear-systems solvers
================================

.. contents:: List of linear solvers
   :local:
   :backlinks: top

.. _compat_potri_bufferSize:

hipsolverDn<type>potri_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotri_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCpotri_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDpotri_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSpotri_bufferSize

.. _compat_potri:

hipsolverDn<type>potri()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotri
   :outline:
.. doxygenfunction:: hipsolverDnCpotri
   :outline:
.. doxygenfunction:: hipsolverDnDpotri
   :outline:
.. doxygenfunction:: hipsolverDnSpotri

.. _compat_potrs:

hipsolverDn<type>potrs()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotrs
   :outline:
.. doxygenfunction:: hipsolverDnCpotrs
   :outline:
.. doxygenfunction:: hipsolverDnDpotrs
   :outline:
.. doxygenfunction:: hipsolverDnSpotrs

.. _compat_potrs_batched:

hipsolverDn<type>potrsBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotrsBatched
   :outline:
.. doxygenfunction:: hipsolverDnCpotrsBatched
   :outline:
.. doxygenfunction:: hipsolverDnDpotrsBatched
   :outline:
.. doxygenfunction:: hipsolverDnSpotrsBatched

.. _compat_getrs:

hipsolverDn<type>getrs()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgetrs
   :outline:
.. doxygenfunction:: hipsolverDnCgetrs
   :outline:
.. doxygenfunction:: hipsolverDnDgetrs
   :outline:
.. doxygenfunction:: hipsolverDnSgetrs

.. _compat_gesv_bufferSize:

hipsolverDn<type><type>gesv_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZZgesv_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCCgesv_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDDgesv_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSSgesv_bufferSize

.. _compat_gesv:

hipsolverDn<type><type>gesv()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZZgesv
   :outline:
.. doxygenfunction:: hipsolverDnCCgesv
   :outline:
.. doxygenfunction:: hipsolverDnDDgesv
   :outline:
.. doxygenfunction:: hipsolverDnSSgesv



.. _compat_leastsqr:

Least-squares solvers
================================

.. contents:: List of least-squares solvers
   :local:
   :backlinks: top

.. _compat_gels_bufferSize:

hipsolverDn<type><type>gels_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZZgels_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCCgels_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDDgels_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSSgels_bufferSize

.. _compat_gels:

hipsolverDn<type><type>gels()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZZgels
   :outline:
.. doxygenfunction:: hipsolverDnCCgels
   :outline:
.. doxygenfunction:: hipsolverDnDDgels
   :outline:
.. doxygenfunction:: hipsolverDnSSgels



.. _compat_eigens:

Symmetric eigensolvers
================================

.. contents:: List of symmetric eigensolvers
   :local:
   :backlinks: top

.. _compat_syevd_bufferSize:

hipsolverDn<type>syevd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsyevd_bufferSize

.. _compat_heevd_bufferSize:

hipsolverDn<type>heevd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCheevd_bufferSize

.. _compat_syevd:

hipsolverDn<type>syevd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevd
   :outline:
.. doxygenfunction:: hipsolverDnSsyevd

.. _compat_heevd:

hipsolverDn<type>heevd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevd
   :outline:
.. doxygenfunction:: hipsolverDnCheevd

.. _compat_sygvd_bufferSize:

hipsolverDn<type>sygvd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsygvd_bufferSize

.. _compat_hegvd_bufferSize:

hipsolverDn<type>hegvd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnChegvd_bufferSize

.. _compat_sygvd:

hipsolverDn<type>sygvd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvd
   :outline:
.. doxygenfunction:: hipsolverDnSsygvd

.. _compat_hegvd:

hipsolverDn<type>hegvd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvd
   :outline:
.. doxygenfunction:: hipsolverDnChegvd



.. _compat_svds:

Singular value decomposition
================================

.. contents:: List of SVD related functions
   :local:
   :backlinks: top

.. _compat_gesvd_bufferSize:

hipsolverDn<type>gesvd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgesvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgesvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgesvd_bufferSize

.. _compat_gesvd:

hipsolverDn<type>gesvd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvd
   :outline:
.. doxygenfunction:: hipsolverDnCgesvd
   :outline:
.. doxygenfunction:: hipsolverDnDgesvd
   :outline:
.. doxygenfunction:: hipsolverDnSgesvd

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

