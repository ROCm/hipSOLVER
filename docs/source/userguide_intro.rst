
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
It sits between the backend library and the user application, marshalling inputs to and outputs from the backend library. Currently, two
backend libraries are supported by hipSOLVER: NVIDIA's `cuSOLVER library <https://developer.nvidia.com/cusolver>`_ and AMD's open-source
`rocSOLVER library <https://github.com/ROCmSoftwarePlatform/rocSOLVER>`_.


Currently implemented functionality
====================================

As with rocSOLVER, the hipSOLVER library remains in active development. New features are being
continuously added, with new functionality documented at each `release of the ROCm platform <https://rocmdocs.amd.com/en/latest/Current_Release_Notes/Current-Release-Notes.html>`_.

The following tables summarize the wrapper functions that are implemented for the different supported precisions in hipSOLVER's latest release.

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


Porting a cuSOLVER application to hipSOLVER
============================================

hipSOLVER is designed to make it easy for users of cuSOLVER to port their applications to hipSOLVER, and provides two separate but interchangeable APIs in order to facilitate
a two-stage transition process. Users are encouraged to start with hipSOLVER's :ref:`compatibility API <library_compat>`, which uses the `hipsolverDn` prefix and has method
signatures that are consistent with cusolverDn functions.

Afterwards, it is recommended to begin the switch to hipSOLVER's :ref:`regular API <library_api>`, which uses the `hipsolver` prefix and introduces minor adjustments to the
API (see below) in order to get the best performance out of the rocSOLVER backend. In most cases, switching to the regular API is as simple as removing `Dn` from the
`hipsolverDn` prefix.

.. _api_differences:

Differences with the cuSOLVER API
----------------------------------

While hipSOLVER's :ref:`compatibility API <library_compat>` and :ref:`regular API <library_api>` are similar to each other, the argument lists of the following functions
differ in the following ways:

* :ref:`hipsolverXXgels_bufferSize <gels_bufferSize>` does not require `dwork` as an argument
* :ref:`hipsolverXXgesv_bufferSize <gesv_bufferSize>` does not require `dwork` as an argument
* :ref:`hipsolverXgesvd_bufferSize <gesvd_bufferSize>` requires `jobu` and `jobv` as arguments
* :ref:`hipsolverXgetrf <getrf>` requires `lwork` as an argument
* :ref:`hipsolverXgetrs <getrs>` requires `work` and `lwork` as arguments
* :ref:`hipsolverXpotrfBatched <potrf_batched>` requires `work` and `lwork` as arguments
* :ref:`hipsolverXpotrs <potrs>` requires `work` and `lwork` as arguments, and
* :ref:`hipsolverXpotrsBatched <potrs_batched>` requires `work` and `lwork` as arguments.

In order to support these changes, the regular API adds the following functions as well:

* :ref:`hipsolverXgetrs_bufferSize <getrs_bufferSize>`
* :ref:`hipsolverXpotrfBatched_bufferSize <potrf_batched_bufferSize>`
* :ref:`hipsolverXpotrs_bufferSize <potrs_bufferSize>`
* :ref:`hipsolverXpotrsBatched_bufferSize <potrs_batched_bufferSize>`

Note that while most hipSOLVER functions take a workspace pointer and size as arguments, rocSOLVER maintains its own internal device workspace by default. In order to take
advantage of this feature, users may pass a null pointer for the `work` argument or a zero size for the `lwork` argument of any function when using the rocSOLVER backend,
and the workspace will be automatically managed behind-the-scenes. It is recommended to use a consistent strategy for workspace management, as performance issues may arise
if the internal workspace is made to flip-flop between user-provided and automatically allocated workspaces, and this feature should not be used with the cuSOLVER backend.

.. _unused_arguments:

Arguments not referenced by rocSOLVER
--------------------------------------

Due to differences in implementation and API design between rocSOLVER and cuSOLVER, certain arguments will not be referenced by the rocSOLVER backend. Keep in mind the
following when using either the API:

* Unlike cuSOLVER, rocSOLVER does not provide information on invalid arguments in its `info` arguments, though it will provide info on singularities and algorithm convergence.
  As a result, the `info` argument of many functions will not be referenced or altered by the rocSOLVER backend, excepting those that provide info on singularities or
  convergence.

* The `niters` argument of :ref:`hipsolverXXgels <gels>` and :ref:`hipsolverXXgesv <gesv>` is not referenced by the rocSOLVER backend.

