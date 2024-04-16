.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _hipsolver:

********************************************************************
hipSOLVER documentation
********************************************************************

hipSOLVER is a LAPACK marshalling library, with multiple supported backends. It sits between the application and a 'worker' LAPACK library, marshalling inputs into the backend library and marshalling results back to the application. hipSOLVER supports rocSOLVER and cuSOLVER as backends. hipSOLVER exports an interface that does not require the client to change, regardless of the chosen backend. 

The code is open and hosted at: https://github.com/ROCm/hipSOLVER

The hipSOLVER documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :ref:`install-linux`

  .. grid-item-card:: How-to

    * :ref:`usage_label`

  .. grid-item-card:: Reference

    * :ref:`api-intro`
    * :ref:`library_api`
    * :ref:`library_compat` 
    * :ref:`library_sparse` 
    * :ref:`library_refactor` 

:ref:`usage_label` is the starting point for new users of the library. For a list of currently implemented routines in the different APIs refer to :ref:`api-intro`. 

To contribute to the documentation refer to `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

