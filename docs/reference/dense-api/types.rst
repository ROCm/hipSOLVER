.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _dense_types:

********************************************************************
Dense matrix datatypes
********************************************************************

hipSOLVER defines types and enumerations that are internally converted to the corresponding backend types at runtime. Here we list the types used in this compatibility API.

hipsolverDnHandle_t
--------------------
.. doxygentypedef:: hipsolverDnHandle_t

hipsolverGesvdjInfo_t
----------------------
See :ref:`hipsolverGesvdjInfo_t <gesvdjinfo_t>`.

hipsolverSyevjInfo_t
--------------------
See :ref:`hipsolverSyevjInfo_t <syevjinfo_t>`.

hipsolverStatus_t
--------------------
See :ref:`hipsolverStatus_t <status_t>`.

hipblasOperation_t
--------------------
See :ref:`hipblasOperation_t <operation_t>`.

hipblasFillMode_t
--------------------
See :ref:`hipblasFillMode_t <fillmode_t>`.

hipblasSideMode_t
--------------------
See :ref:`hipblasSideMode_t <sidemode_t>`.

hipsolverEigMode_t
--------------------
See :ref:`hipsolverEigMode_t <eigmode_t>`.

hipsolverEigType_t
--------------------
See :ref:`hipsolverEigType_t <eigtype_t>`.

hipsolverEigRange_t
--------------------
.. doxygenenum:: hipsolverEigRange_t

hipsolverAlgMode_t
--------------------
.. doxygenenum:: hipsolverAlgMode_t

hipsolverDnFunction_t
---------------------
.. doxygenenum:: hipsolverDnFunction_t

