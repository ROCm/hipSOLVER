
*************
Introduction
*************

.. toctree::
   :maxdepth: 4

.. contents:: Table of contents
   :local:
   :backlinks: top


Library overview
==========================

hipSOLVER is an open-source marshalling library for `LAPACK routines <https://www.netlib.org/lapack/explore-html/modules.html>`_ on the GPU.
It sits between a backend library and the user application, marshalling inputs to and outputs from the backend library so that the user
application remains unchanged when using different backends. Currently, two backend libraries are supported by hipSOLVER: NVIDIA's `cuSOLVER
library <https://developer.nvidia.com/cusolver>`_ and AMD's open-source `rocSOLVER library <https://github.com/ROCmSoftwarePlatform/rocSOLVER>`_.

The :ref:`regular hipSOLVER API <library_api>` is a thin wrapper layer around the different backends. As such, it is not expected to introduce
significant overhead. However, its main purpose is portability, so when performance is critical, directly using the library backend corresponding
to the given architecture is recommended.

Another purpose of hipSOLVER is to facilitate the translation of cuSOLVER applications to
`AMD's open source ROCm platform <https://rocmdocs.amd.com/en/latest/index.html>`_ ecosystem. To that end, hipSOLVER also includes a
:ref:`compatibility API <library_compat>` with method signatures that match exactly those of cuSOLVER. For more details see the
section :ref:`usage_label`.


Currently implemented functionality
====================================

The hipSOLVER library remains in active development. New features are being continuously added, with new functionality documented at each
`release of the ROCm platform <https://rocmdocs.amd.com/en/latest/Current_Release_Notes/Current-Release-Notes.html>`_.

The following tables summarize the wrapper functions that are implemented in the regular API for the different supported precisions in
hipSOLVER's latest release. Most of these functions have a corresponding version in the compatibility API, where applicable.

LAPACK auxiliary functions
----------------------------

.. csv-table:: Orthonormal matrices
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXorgbr_bufferSize <orgbr_bufferSize>`, x, x, ,
    :ref:`hipsolverXorgbr <orgbr>`, x, x, ,
    :ref:`hipsolverXorgqr_bufferSize <orgqr_bufferSize>`, x, x, ,
    :ref:`hipsolverXorgqr <orgqr>`, x, x, ,
    :ref:`hipsolverXorgtr_bufferSize <orgtr_bufferSize>`, x, x, ,
    :ref:`hipsolverXorgtr <orgtr>`, x, x, ,
    :ref:`hipsolverXormqr_bufferSize <ormqr_bufferSize>`, x, x, ,
    :ref:`hipsolverXormqr <ormqr>`, x, x, ,
    :ref:`hipsolverXormtr_bufferSize <ormtr_bufferSize>`, x, x, ,
    :ref:`hipsolverXormtr <ormtr>`, x, x, ,

.. csv-table:: Unitary matrices
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXungbr_bufferSize <ungbr_bufferSize>`, , , x, x
    :ref:`hipsolverXungbr <ungbr>`, , , x, x
    :ref:`hipsolverXungqr_bufferSize <ungqr_bufferSize>`, , , x, x
    :ref:`hipsolverXungqr <ungqr>`, , , x, x
    :ref:`hipsolverXungtr_bufferSize <ungtr_bufferSize>`, , , x, x
    :ref:`hipsolverXungtr <ungtr>`, , , x, x
    :ref:`hipsolverXunmqr_bufferSize <unmqr_bufferSize>`, , , x, x
    :ref:`hipsolverXunmqr <unmqr>`, , , x, x
    :ref:`hipsolverXunmtr_bufferSize <unmtr_bufferSize>`, , , x, x
    :ref:`hipsolverXunmtr <unmtr>`, , , x, x

LAPACK main functions
----------------------------

.. csv-table:: Triangular factorizations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXpotrf_bufferSize <potrf_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotrf <potrf>`, x, x, x, x
    :ref:`hipsolverXpotrfBatched_bufferSize <potrf_batched_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotrfBatched <potrf_batched>`, x, x, x, x
    :ref:`hipsolverXgetrf_bufferSize <getrf_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgetrf <getrf>`, x, x, x, x
    :ref:`hipsolverXsytrf_bufferSize <sytrf_bufferSize>`, x, x, x, x
    :ref:`hipsolverXsytrf <sytrf>`, x, x, x, x

.. csv-table:: Orthogonal factorizations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXgeqrf_bufferSize <geqrf_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgeqrf <geqrf>`, x, x, x, x

.. csv-table:: Problem and matrix reductions
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXsytrd_bufferSize <sytrd_bufferSize>`, x, x, ,
    :ref:`hipsolverXsytrd <sytrd>`, x, x, ,
    :ref:`hipsolverXhetrd_bufferSize <hetrd_bufferSize>`, , , x, x
    :ref:`hipsolverXhetrd <hetrd>`, , , x, x
    :ref:`hipsolverXgebrd_bufferSize <gebrd_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgebrd <gebrd>`, x, x, x, x

.. csv-table:: Linear-systems solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXpotri_bufferSize <potri_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotri <potri>`, x, x, x, x
    :ref:`hipsolverXpotrs_bufferSize <potrs_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotrs <potrs>`, x, x, x, x
    :ref:`hipsolverXpotrsBatched_bufferSize <potrs_batched_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotrsBatched <potrs_batched>`, x, x, x, x
    :ref:`hipsolverXgetrs_bufferSize <getrs_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgetrs <getrs>`, x, x, x, x
    :ref:`hipsolverXXgesv_bufferSize <gesv_bufferSize>`, x, x, x, x
    :ref:`hipsolverXXgesv <gesv>`, x, x, x, x

.. csv-table:: Least-square solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXXgels_bufferSize <gels_bufferSize>`, x, x, x, x
    :ref:`hipsolverXXgels <gels>`, x, x, x, x

.. csv-table:: Symmetric eigensolvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXsyevd_bufferSize <syevd_bufferSize>`, x, x, ,
    :ref:`hipsolverXsyevd <syevd>`, x, x, ,
    :ref:`hipsolverXsygvd_bufferSize <sygvd_bufferSize>`, x, x, ,
    :ref:`hipsolverXsygvd <sygvd>`, x, x, ,
    :ref:`hipsolverXheevd_bufferSize <heevd_bufferSize>`, , , x, x
    :ref:`hipsolverXheevd <heevd>`, , , x, x
    :ref:`hipsolverXhegvd_bufferSize <hegvd_bufferSize>`, , , x, x
    :ref:`hipsolverXhegvd <hegvd>`, , , x, x

.. csv-table:: Singular value decomposition
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXgesvd_bufferSize <gesvd_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgesvd <gesvd>`, x, x, x, x

LAPACK-like functions
----------------------------

.. csv-table:: Symmetric eigensolvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXsyevj_bufferSize <syevj_bufferSize>`, x, x, ,
    :ref:`hipsolverXsyevj <syevj>`, x, x, ,
    :ref:`hipsolverXsyevjBatched_bufferSize <syevj_batched_bufferSize>`, x, x, ,
    :ref:`hipsolverXsyevjBatched <syevj_batched>`, x, x, ,
    :ref:`hipsolverXsygvj_bufferSize <sygvj_bufferSize>`, x, x, ,
    :ref:`hipsolverXsygvj <sygvj>`, x, x, ,
    :ref:`hipsolverXheevj_bufferSize <heevj_bufferSize>`, , , x, x
    :ref:`hipsolverXheevj <heevj>`, , , x, x
    :ref:`hipsolverXheevjBatched_bufferSize <heevj_batched_bufferSize>`, , , x, x
    :ref:`hipsolverXheevjBatched <heevj_batched>`, , , x, x
    :ref:`hipsolverXhegvj_bufferSize <hegvj_bufferSize>`, , , x, x
    :ref:`hipsolverXhegvj <hegvj>`, , , x, x

.. csv-table:: Singular value decomposition
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverDnXgesvdj_bufferSize <compat_gesvdj_bufferSize>`, x, x, x, x
    :ref:`hipsolverDnXgesvdj <compat_gesvdj>`, x, x, x, x
    :ref:`hipsolverDnXgesvdjBatched_bufferSize <compat_gesvdj_batched_bufferSize>`, x, x, x, x
    :ref:`hipsolverDnXgesvdjBatched <compat_gesvdj_batched>`, x, x, x, x


Compatibility-only functions
====================================

The following tables summarize the wrapper functions that have not been implemented in rocSOLVER, but can be computed with an equivalent function
that uses a different algorithmic approach. These functions are provided as part of the compatibility API only, pending their implementation
in rocSOLVER.

Partial eigensolver functions
------------------------------

Partial eigensolvers have been implemented in rocSOLVER, but at present they do not use a divide-and-conquer approach.

.. csv-table:: Symmetric eigensolvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverDnXsyevdx_bufferSize <compat_syevdx_bufferSize>`, x, x, ,
    :ref:`hipsolverDnXsyevdx <compat_syevdx>`, x, x, ,
    :ref:`hipsolverDnXsygvdx_bufferSize <compat_sygvdx_bufferSize>`, x, x, ,
    :ref:`hipsolverDnXsygvdx <compat_sygvdx>`, x, x, ,
    :ref:`hipsolverDnXheevdx_bufferSize <compat_heevdx_bufferSize>`, , , x, x
    :ref:`hipsolverDnXheevdx <compat_heevdx>`, , , x, x
    :ref:`hipsolverDnXhegvdx_bufferSize <compat_hegvdx_bufferSize>`, , , x, x
    :ref:`hipsolverDnXhegvdx <compat_hegvdx>`, , , x, x

Partial SVD functions
------------------------------

Partial SVD has been implemented in rocSOLVER, but at present it does not use an approximate algorithm, nor does it compute the residual norm.

.. csv-table:: Singular value decomposition
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverDnXgesvdaStridedBatched_bufferSize <compat_gesvda_strided_batched_bufferSize>`, x, x, x, x
    :ref:`hipsolverDnXgesvdaStridedBatched <compat_gesvda_strided_batched>`, x, x, x, x

