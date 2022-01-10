.. _usage_label:

*****************
Using hipSOLVER
*****************

Once installed, hipSOLVER can be used just like any other library with a C API. 
The header file will need to be included in the user code, and the shared library will become link-time and run-time dependencies 
for the user application. The user code can be ported, with no changes, to any system with hipSOLVER installed regardless of the
backend library. 

For more details on how to use the API methods, see the code samples at the 
`hipSOLVER's github page <https://github.com/ROCmSoftwarePlatform/hipSOLVER/tree/develop/clients/samples>`_, or
the documentation of the corresponding backend libraries.  

.. toctree::
   :maxdepth: 4

.. contents:: Table of contents
   :local:
   :backlinks: top


.. _api_differences:

Some considerations when Using the hipSOLVER regular API
==========================================================

The hipSOLVER regular API is similar to cuSOLVER; however, due to differences in the implementation and design between 
cuSOLVER and rocSOLVER, some minor adjustements were introduced to ensure the best performance out of either backend. 


Different signatures and additional API methods
------------------------------------------------

- The methods to obtain the size of the workspace needed by functions `gels` and `gesv` in cuSOLVER require `dwork` as
  and argument, however, it is never used and can be null. On the rocSOLVER side, `dwork` is not needed to calculate the workspace size. In consequence:
  
  * :ref:`hipsolverXXgels_bufferSize <gels_bufferSize>` does not require `dwork` as an argument, and 
  * :ref:`hipsolverXXgesv_bufferSize <gesv_bufferSize>` does not require `dwork` as an argument. 
  
  (These wrappers pass `dwork = nullptr` when calling cuSOLVER). 

- To calculate the workspace required by function `gesvd` in rocSOLVER, the values of `jobu` and `jobv` are needed. As a result, 
  
  * :ref:`hipsolverXgesvd_bufferSize <gesvd_bufferSize>` requires `jobu` and `jobv` as arguments.

  (These arguments are simply ignored when the wrapper calls cuSOLVER as they are not needed). 

- To properly use the user-provided workspace, rocSOLVER requires both, the allocated pointer and its size. Consequently:

  * :ref:`hipsolverXgetrf <getrf>` requires `lwork` as an argument.

  (`lwork` is simply ignored when the wrapper calls cuSOLVER as it is not needed).

- All the rocSOLVER functions called by hipSOLVER require a workspace. To allow the user to specify one, 

  * :ref:`hipsolverXgetrs <getrs>` requires `work` and `lwork` as arguments,
  * :ref:`hipsolverXpotrfBatched <potrf_batched>` requires `work` and `lwork` as arguments,
  * :ref:`hipsolverXpotrs <potrs>` requires `work` and `lwork` as arguments, and
  * :ref:`hipsolverXpotrsBatched <potrs_batched>` requires `work` and `lwork` as arguments.

  (These arguments are simply ignored when these wrappers call cuSOLVER as they are not needed).

  In order to support these changes, the regular API adds the following functions as well:

  * :ref:`hipsolverXgetrs_bufferSize <getrs_bufferSize>`
  * :ref:`hipsolverXpotrfBatched_bufferSize <potrf_batched_bufferSize>`
  * :ref:`hipsolverXpotrs_bufferSize <potrs_bufferSize>`
  * :ref:`hipsolverXpotrsBatched_bufferSize <potrs_batched_bufferSize>`

  (These methods return `lwork = 0` when using the cuSOLVER backend as the corresponding functions
  in cuSOLVER do not need workspace).


Arguments not referenced by rocSOLVER
--------------------------------------

- Unlike cuSOLVER, rocSOLVER does not provide information on invalid arguments in its `info` arguments, though it will provide info on singularities and algorithm convergence.
  As a result, the `info` argument of many functions will not be referenced or altered by the rocSOLVER backend, excepting those that provide info on singularities or convergence.

- The `niters` argument of :ref:`hipsolverXXgels <gels>` and :ref:`hipsolverXXgesv <gesv>` is not referenced by the rocSOLVER backend; there is no iterative refinement
  currently implemented in rocSOLVER. 


.. _mem_model:

Using rocSOLVER's memory model
---------------------------------

Most hipSOLVER functions take a workspace pointer and size as arguments, allowing the user to manage the device memory used internally by the backends. 
rocSOLVER, however, can maintain the device workspace automatically by default 
(see `rocSOLVER's memory model <https://rocsolver.readthedocs.io/en/master/userguide_memory.html>`_ for more details). In order to take
advantage of this feature, users may pass a null pointer for the `work` argument or a zero size for the `lwork` argument of any function when using the rocSOLVER backend,
and the workspace will be automatically managed behind-the-scenes. It is recommended, however, to use a consistent strategy for workspace management, as performance issues may arise
if the internal workspace is made to flip-flop between user-provided and automatically allocated workspaces. 

.. warning::
    This feature should not be used with the cuSOLVER backend; hipSOLVER does not guarantee a defined behavior when passing 
    a null workspace to cuSOLVER functions that require one.


Using rocSOLVER's in-place functions
--------------------------------------

The solvers `gesv` and `gels` in cuSOLVER are out-of-place in the sense that the solution vectors `X` do not overwrite the
input matrix `B`. In rocSOLVER this is not the case; when `hipsolverXXgels` or `hipsolverXXgesv` call rocSOLVER, some data
movements must be done internally to restore `B` and copy the results back to `X`. These copies could introduce a noticeable overhead depending
on the size of the matrices. To avoid this potential problem, the users can pass `X = B` to 
`hipsolverXXgels` or `hipsolverXXgesv` when using the rocSOLVER backend; in this case, no data movements will be required, and the
solution vectors can be retrieved using either `B` or `X`.

.. warning::
    This feature should not be used with the cuSOLVER backend; hipSOLVER does not guarantee a defined behavior when passing
    `X = B` to the mentioned functions in cuSOLVER.


.. _porting:

Porting cuSOLVER applications to hipSOLVER
============================================

hipSOLVER is also designed to make it easy for users of cuSOLVER to port their existing applications to hipSOLVER, and provides two separate but interchangeable APIs in order to facilitate
a two-stage transition process. Users are encouraged to start with hipSOLVER's :ref:`compatibility API <library_compat>`, which uses the `hipsolverDn` prefix and has method
signatures that are fully consistent with cusolverDn functions. As explained below, however, the compatibility API may introduce some performance drawbacks, especially when using the
rocSOLVER backend. So, as a second stage, it is recommended to begin the switch to hipSOLVER's :ref:`regular API <library_api>`, which uses the `hipsolver` prefix and introduces minor adjustments to the
API (see section :ref:`1.3.1 <api_differences>`) in order to get the best performance out of the rocSOLVER backend. In most cases, switching to the regular API is as simple as removing `Dn` from the
`hipsolverDn` prefix.

(No matter which API is used, a hipSOLVER application can be executed, without modifications to the code, in systems with cuSOLVER or rocSOLVER installed. However, using the
regular API ensures the best performance out of either backend).


.. _porting_issues:

Possible performance implications of the compatibility API
------------------------------------------------------------

- To calculate the workspace required by function `gesvd` in rocSOLVER, the values of `jobu` and `jobv` are needed, however,
  the function :ref:`hipsolverDnXgesvd_bufferSize <compat_gesvd_bufferSize>` does not accept these arguments. So, when using
  the rocSOLVER backend, `hipsolverDnXgesvd_bufferSize` has to calculate internally the workspace for all possible values of `jobu` and `jobv`,
  and return the maximum. 

  (`hipsolverDnXgesvd_bufferSize` is slower than `hipsolverXgesvd_bufferSize`, and its returned workspace size could be slightly larger than what is actually needed).

- To properly use the user-provided workspace, rocSOLVER requires both, the allocated pointer and its size, however, the function 
  :ref:`hipsolverDnXgetrf <compat_getrf>` does not accept `lwork` as an argument. In consequence, when using the rocSOLVER backend,
  `hipsolverDnXgetrf` has to call internally `hipsolverDnXgetrf_bufferSize` to know the size of the workspace.

  (`hipsolverDnXgetrf_bufferSize` gets to be called twice in practice, once by the user before allocating the workspace, and once 
  by hipSOLVER internally when executing `hipsolverDnXgetrf` function. `hipsolverDnXgetrf` could be slightly slower than `hipsolverXgetrf`
  because of the extra call to the buffersize helper).

- The functions :ref:`hipsolverDnXgetrs <compat_getrs>`, :ref:`hipsolverDnXpotrs <compat_potrs>`, :ref:`hipsolverDnXpotrsBatched <compat_potrs_batched>`, and
  :ref:`hipsolverDnXpotrfBatched <compat_potrf_batched>` do not accept `work` and `lwork` as arguments, however, this functionality does require a non-zero workspace
  in rocSOLVER. As a result, when using the rocSOLVER backend, these functions will switch to automatic workspace managment model (see :ref:`here <mem_model>`).

  (Users must keep in mind that even if the compatibility API does not have bufferSize helpers for the mentioned functions, these functions do require 
  workspace when using rocSOLVER, and it will be automatically managed. This may imply device memory reallocations with its corresponding overheads).

