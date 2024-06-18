.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _install-linux:

*****************************
Installation on Linux
*****************************

Install pre-built packages
===========================

Download pre-built packages from `ROCm's package servers <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html>`_. Updates to each release are listed in the ``CHANGELOG.md`` file under the releases tab of the `hipSOLVER github page <https://github.com/ROCm/hipSOLVER>`_.

* `sudo apt update && sudo apt install hipsolver`

.. note::
    The pre-built packages depend on the third-party library SuiteSparse, which must be installed on the system prior to installing hipSOLVER. SuiteSparse can be installed using the package manager of most distros.


Build & install library using script (Ubuntu only)
===================================================

The root of the `hipSOLVER repository <https://github.com/ROCm/hipSOLVER>`_ has a helper bash script ``install.sh`` to build and install hipSOLVER on Ubuntu with a single command.  It does not take a lot of options and hard-codes configuration that can be specified through invoking ``cmake`` directly, but it's a great way to get started quickly and can serve as an example of how to build/install.  A few commands in the script need sudo access,
so it may prompt you for a password.

* ``./install.sh -id`` --- build library, build dependencies, and install (-d flag only needs to be passed once on a system).
* ``./install.sh -ic`` --- build library, build clients (tests, benchmarks, and samples), and install.
* ``./install.sh --cuda`` --- build library on a CUDA-enabled machine, with cuSOLVER as the backend.
* ``./install.sh --no-sparse`` --- build library without hipsolverSp functionality, with rocSOLVER as the backend.

To see more options, use the help option of the install script.

* ``./install.sh -h``


Build & install library manually
=================================

For a standard library installation, follow these steps:

.. code-block:: bash

    mkdir -p <HIPSOLVER_BUILD_DIR_PATH>/release
    cd <HIPSOLVER_BUILD_DIR_PATH>/release
    CXX=/opt/rocm/bin/hipcc cmake <HIPSOLVER_SOURCE_DIR_PATH>
    make -j$(nproc)
    sudo make install

sudo is required if installing into a system directory such as /opt/rocm, which is the default option.

* Use ``-DCMAKE_INSTALL_PREFIX=<other_path>`` to specify a different install directory.
* Use ``-DCMAKE_BUILD_TYPE=<other_configuration>`` to specify a build configuration, such as 'Debug'. The default build configuration is 'Release'.

Library dependencies
---------------------

The hipSOLVER library has two separate sets of dependencies, depending on the desired backend. The cuSOLVER backend has the following dependencies:

1. cuSOLVER

The rocSOLVER backend has the following dependencies:

1. `rocSOLVER <https://github.com/ROCmSoftwarePlatform/rocSOLVER>`_
2. `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_
3. `rocSPARSE <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_ (optional, required by default)
4. `SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_ modules CHOLMOD and SuiteSparse_config (optional, required by default)

rocSOLVER itself depends on rocBLAS and rocSPARSE, therefore all three libraries should be present with a standard rocSOLVER installation. For more information
about building and installing rocSOLVER, refer to the :doc:`rocSOLVER documentation <rocsolver:installation/installlinux>`.

SuiteSparse is a third-party library, and can be installed using the package managers of most distros. Together with rocSPARSE, it is used to provide
functionality for the hipsolverSp API. If only hipsolverDn and/or hipsolverRf are needed, these dependencies can be ignored by setting the ``BUILD_WITH_SPARSE``
option to ``OFF``.

* ``DBUILD_WITH_SPARSE=OFF``


Build library + tests + benchmarks + samples manually
======================================================

The repository contains source code for client programs that serve as tests, benchmarks, and samples. Client source code can be found in the clients subdirectory.

Client dependencies
--------------------

The hipSOLVER samples have no external dependencies, but our unit test and benchmarking applications do. These clients introduce the following dependencies:

1. `lapack <https://github.com/Reference-LAPACK/lapack-release>`_ (lapack itself brings a dependency on a fortran compiler)
2. `googletest <https://github.com/google/googletest>`_
3. `hipBLAS <https://github.com/ROCm/hipBLAS>`_ (optional)
4. `hipSPARSE <https://github.com/ROCm/hipSPARSE>`_ (optional, required by default)

Unfortunately, many distros do not provide a googletest package with pre-compiled libraries, and the lapack packages do not have the necessary cmake config files for cmake to configure linking the cblas library. hipSOLVER provides a cmake script that builds
lapack and googletest from source. This is an optional step; users can provide their own builds of these dependencies and help cmake find them by setting
the ``CMAKE_PREFIX_PATH`` definition. The following is a sequence of steps to build dependencies and install them to the cmake default, /usr/local:

.. code-block:: bash

    mkdir -p <HIPSOLVER_BUILD_DIR_PATH>/release/deps
    cd <HIPSOLVER_BUILD_DIR_PATH>/release/deps
    cmake -DBUILD_BOOST=OFF <HIPSOLVER_SOURCE_PATH>/deps   #assuming boost is installed through package manager as above
    make -j$(nproc) install

hipBLAS is only required if the ``BUILD_HIPBLAS_TESTS`` option is set to ``ON``, and is used to ensure compatibility between the hipblas enums defined
separately by hipBLAS and hipSOLVER. hipSPARSE is required by default but can be ignored if the ``BUILD_WITH_SPARSE`` option is set to ``OFF``, and is used
to create objects required by tests for the hipsolverSp API.

* ``DBUILD_HIPBLAS_TESTS=ON``
* ``DBUILD_WITH_SPARSE=OFF``

Both libraries can be installed similarly to hipSOLVER. For example, the install scripts for hipBLAS and hipSPARSE can each be invoked to build and
install the respective library via:

* ``./install.sh -i``

More details can be found in the `hipBLAS documentation <https://rocm.docs.amd.com/projects/hipBLAS/en/latest/index.html>`_
and the `hipSPARSE documentation <https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/index.html>`_.

Library and clients
--------------------

Once dependencies are available on the system, it is possible to configure the clients to build. This requires a few extra cmake flags to the library's
cmake configure script. If the dependencies are not installed into system defaults (like /usr/local), you should pass the ``CMAKE_PREFIX_PATH`` to cmake
to help find them.

* ``-DCMAKE_PREFIX_PATH="<semicolon separated paths>"``

.. code-block:: bash

    CXX=/opt/rocm/bin/hipcc cmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON [HIPSOLVER_SOURCE]
    make -j$(nproc)
    sudo make install   # sudo required if installing into system directory such as /opt/rocm
