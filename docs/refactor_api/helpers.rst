.. _refactor_helpers:

****************
Helper Functions
****************

These are helper functions that control aspects of the hipSOLVER library. These are divided
into four categories:

* :ref:`refactor_initialize` functions. Used to initialize and cleanup the library handle.
* :ref:`refactor_input` functions. Provide functionality to manipulate function input.
* :ref:`refactor_output` functions. Provide functionality to access function output.
* :ref:`refactor_parameters` functions. Provide functionality to manipulate parameters.


.. _refactor_initialize:

Handle set-up and tear-down
===============================

.. contents:: List of handle initialization functions
   :local:
   :backlinks: top

hipsolverRfCreate()
-----------------------------------------
.. doxygenfunction:: hipsolverRfCreate

hipsolverRfDestroy()
-----------------------------------------
.. doxygenfunction:: hipsolverRfDestroy



.. _refactor_input:

Input manipulation
===============================

.. contents:: List of input functions
   :local:
   :backlinks: top

hipsolverRfSetupDevice()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetupDevice

hipsolverRfSetupHost()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetupHost

hipsolverRfBatchSetupHost()
-----------------------------------------
.. doxygenfunction:: hipsolverRfBatchSetupHost

hipsolverRfAnalyze()
-----------------------------------------
.. doxygenfunction:: hipsolverRfAnalyze

hipsolverRfBatchAnalyze()
-----------------------------------------
.. doxygenfunction:: hipsolverRfBatchAnalyze

hipsolverRfResetValues()
-----------------------------------------
.. doxygenfunction:: hipsolverRfResetValues

hipsolverRfBatchResetValues()
-----------------------------------------
.. doxygenfunction:: hipsolverRfBatchResetValues



.. _refactor_output:

Output manipulation
===============================

.. contents:: List of output functions
   :local:
   :backlinks: top

hipsolverRfAccessBundledFactorsDevice()
-----------------------------------------
.. doxygenfunction:: hipsolverRfAccessBundledFactorsDevice

hipsolverRfExtractBundledFactorsHost()
-----------------------------------------
.. doxygenfunction:: hipsolverRfExtractBundledFactorsHost

hipsolverRfExtractSplitFactorsHost()
-----------------------------------------
.. doxygenfunction:: hipsolverRfExtractSplitFactorsHost

hipsolverRfBatchZeroPivot()
-----------------------------------------
.. doxygenfunction:: hipsolverRfBatchZeroPivot



.. _refactor_parameters:

Parameter manipulation
===============================

.. contents:: List of parameter functions
   :local:
   :backlinks: top

hipsolverRfGet_Algs()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGet_Algs

hipsolverRfGetMatrixFormat()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGetMatrixFormat

hipsolverRfGetNumericBoostReport()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGetNumericBoostReport

hipsolverRfGetNumericProperties()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGetNumericProperties

hipsolverRfGetResetValuesFastMode()
-----------------------------------------
.. doxygenfunction:: hipsolverRfGetResetValuesFastMode

hipsolverRfSetAlgs()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetAlgs

hipsolverRfSetMatrixFormat()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetMatrixFormat

hipsolverRfSetNumericProperties()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetNumericProperties

hipsolverRfSetResetValuesFastMode()
-----------------------------------------
.. doxygenfunction:: hipsolverRfSetResetValuesFastMode

