
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


.. _api_differences:

Differences with the cuSOLVER API
==================================

While the API of hipSOLVER is similar to that of cuSOLVER, there are some notable differences. In particular:

* :ref:`hipsolverXgesvd_bufferSize <gesvd_bufferSize>` requires `jobu` and `jobv` as arguments
* :ref:`hipsolverXgetrf <getrf>` requires `lwork` as an argument
* :ref:`hipsolverXgetrs <getrs>` requires `work` and `lwork` as arguments
* :ref:`hipsolverXpotrfBatched <potrf_batched>` requires `work` and `lwork` as arguments
* :ref:`hipsolverXpotrs <potrs>` requires `work` and `lwork` as arguments, and
* :ref:`hipsolverXpotrsBatched <potrs_batched>` requires `work` and `lwork` as arguments.

In order to support these changes, hipSOLVER adds the following functions as well:

* :ref:`hipsolverXgetrs_bufferSize <getrs_bufferSize>`
* :ref:`hipsolverXpotrfBatched_bufferSize <potrf_batched_bufferSize>`
* :ref:`hipsolverXpotrs_bufferSize <potrs_bufferSize>`
* :ref:`hipsolverXpotrsBatched_bufferSize <potrs_batched_bufferSize>`

Furthermore, due to differences in implementation and API design between rocSOLVER and cuSOLVER, not all arguments are handled identically between the two backends.
When using the rocSOLVER backend, keep in mind the following differences:

* While many cuSOLVER and hipSOLVER functions take a workspace pointer and size as arguments, rocSOLVER maintains its own internal device
  workspace by default. In order to take advantage of this feature, users may pass a null pointer for the `work` argument of any function when using the rocSOLVER backend,
  and the workspace will be automatically managed behind-the-scenes. It is recommended to use a consistent strategy for workspace management, as performance issues may arise
  if the internal workspace is made to flip-flop between user-provided and automatically allocated workspaces.

* Additionally, unlike cuSOLVER, rocSOLVER does not provide information on invalid arguments in its `info` arguments, though it will provide info on singularities and
  algorithm convergence. As a result, the `info` argument of many functions will not be referenced or altered by the rocSOLVER backend, excepting those that provide info on
  singularities or convergence.


