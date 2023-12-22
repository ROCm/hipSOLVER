.. _library_api:

##############################################
hipSOLVER Regular API
##############################################

Currently, this API document only provides the method signatures for the wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backends' documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/>`_ and/or `rocSOLVER API <https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/api/index.html>`_.

The hipSOLVER API is designed to be similar to the cusolver and rocSOLVER interfaces, but it requires some minor adjustments to ensure
the best performance out of both backends. Generally, this involves the addition of workspace parameters and some additional API methods.
Please refer to the user guide for a complete listing of :ref:`API differences <api_differences>`.

Users interested in using hipSOLVER without these adjustments, so that the interface matches cuSOLVER, should instead consult the
:ref:`Compatibility API documentation <library_compat>`. See also :ref:`the porting section <porting>` for more details.

.. tableofcontents::
