
*************
Installation
*************


.. toctree::
   :maxdepth: 4

.. contents:: Table of contents
   :local:
   :backlinks: top


Install pre-built packages
===========================

Download pre-built packages from `ROCm's package servers <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_. Release notes
are available on the releases tab of the `library's github page <https://github.com/ROCmSoftwarePlatform/hipSOLVER>`_.

* `sudo apt update && sudo apt install hipsolver`


Build & install library using script (Ubuntu only)
===================================================

The root of the `hipSOLVER repository <https://github.com/ROCmSoftwarePlatform/hipSOLVER>`_ has a helper bash script ``install.sh`` to build and install
hipSOLVER on Ubuntu with a single command.  It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake
directly, but it's a great way to get started quickly and can serve as an example of how to build/install.  A few commands in the script need sudo access,
so it may prompt you for a password.

* ``./install.sh -id`` --- build library, build dependencies, and install (-d flag only needs to be passed once on a system).
* ``./install.sh -ic`` --- build library, build clients (tests, benchmarks, and samples), and install.

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


Build library + tests + benchmarks + samples manually
======================================================

The repository contains source code for client programs that serve as tests, benchmarks, and samples. Client source code can be found in the clients subdirectory.

Dependencies (only necessary for hipSOLVER clients)
----------------------------------------------------

The hipSOLVER samples have no external dependencies, but our unit test and benchmarking applications do. These clients introduce the following dependencies:

1. `lapack <https://github.com/Reference-LAPACK/lapack-release>`_ (lapack itself brings a dependency on a fortran compiler)
2. `googletest <https://github.com/google/googletest>`_

Unfortunately, many distros do not provide a googletest package with pre-compiled libraries, and the
lapack packages do not have the necessary cmake config files for cmake to configure linking the cblas library. hipSOLVER provide a cmake script that builds
the above dependencies from source. This is an optional step; users can provide their own builds of these dependencies and help cmake find them by setting
the ``CMAKE_PREFIX_PATH`` definition. The following is a sequence of steps to build dependencies and install them to the cmake default, /usr/local:

.. code-block:: bash

    mkdir -p <HIPSOLVER_BUILD_DIR_PATH>/release/deps
    cd <HIPSOLVER_BUILD_DIR_PATH>/release/deps
    cmake <HIPSOLVER_SOURCE_PATH>/deps
    make -j$(nproc) install

Library and clients
--------------------

Once dependencies are available on the system, it is possible to configure the clients to build. This requires a few extra cmake flags to the library's
cmake configure script. If the dependencies are not installed into system defaults (like /usr/local), you should pass the ``CMAKE_PREFIX_PATH`` to cmake
to help find them.

* ``-DCMAKE_PREFIX_PATH="<semicolon separated paths>"``

.. code-block:: bash

    CXX=/opt/rocm/bin/hcc cmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON [HIPSOLVER_SOURCE]
    make -j$(nproc)
    sudo make install   # sudo required if installing into system directory such as /opt/rocm
