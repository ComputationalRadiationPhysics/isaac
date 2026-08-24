![ISAAC](/isaac.png?raw=true "ISAAC")

ISAAC Install Guide
=====================================================

Requirements
------------

Most dependencies are part of most distributions. However some need to
be built yourself nevertheless or the distribution versions are outdated.

### Requirements for the server and the in situ library

* __gcc__ / __g++__ for compiling the program at all. At least version 10
  should be used:
  * _Debian/Ubuntu_:
    * `sudo apt-get install gcc-10 g++-10 build-essential`
* __CMake__ 3.25.2+ for building everything:
  * _Debian/Ubuntu_:
    * `sudo apt-get install cmake cmake-curses-gui`
  * _From Source_:
    * `wget https://cmake.org/files/v3.28/cmake-3.28.6.tar.gz`
    * `tar -xzf cmake-3.28.6.tar.gz`
    * `rm cmake-3.28.6.tar.gz`
    * `cd cmake-3.28.6`
    * With admin rights and no other version of cmake installed:
      * `./configure`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `./configure --prefix=$CMAKE_INSTALL_DIR`, where `$CMAKE_INSTALL_DIR`
      is the full path to the directory where cmake-3.28.6 should be installed.
      * `make install`
      * Now, a local version of CMake is installed in the specified install
        directory. Later, while compiling an application using CMake (including
        the ISAAC server and the ISAAC examples), use `$CMAKE_INSTALL_DIR/bin/cmake`
        instead of `cmake` and `$CMAKE_INSTALL_DIR/bin/ccmake` instead of
        `ccmake`, where `$CMAKE_INSTALL_DIR` is the path of the cmake-3.28.6
        install directory used above.
* __libjpeg-turbo__ for (de)compressing the rendered image of the transmission:
  * _Debian/Ubuntu_:
    * `sudo apt-get install libjpeg-dev`
  * _From Source_:
    * You might need to install [__nasm__](https://nasm.us/), as building 
      libjpeg-turbo requires it.
    * `git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git`
    * `cd libjpeg-turbo`
    * `mkdir build && cd build`
    * With admin rights and no other version of libjpeg-turbo installed:
      * `cmake ..`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `cmake .. -DCMAKE_INSTALL_PREFIX=$LIBJPEG_INSTALL_DIR`
      * `make install`
* __Jansson__ 2.12+ for the de- and encryption of the JSON messages transfered
  between server and client.
  * _Debian/Ubuntu_:
    * `sudo apt-get install libjansson-dev`
  * _From Source_:
    * `git clone https://github.com/akheron/jansson.git`
    * `cd jansson`
    * `mkdir build && cd build`
    * With admin rights and no other version of jansson installed:
      * `cmake ..`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `cmake .. -DCMAKE_INSTALL_PREFIX=$JANSSON_INSTALL_DIR`
      * `make install`
      * Now, a local version of Jansson is installed in the specified install
        directory. Later, while compiling an application using Jansson
        (including the ISAAC server and the ISAAC examples), add
        `-Djansson_DIR=$JANSSON_INSTALL_DIR/lib/cmake/jansson`, where
        `$JANSSON_INSTALL_DIR` is the path of the Jansson install directory
        used above.
* __Boost__ 1.70+ is needed, but only template libraries, so no
  system wide installation or static linking is needed here:
  * _Debian/Ubuntu_:
    * `sudo apt-get install libboost-dev`
  * _From Source_:
    * `wget https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz`
    * `tar -xzf boost_1_87_0.tar.gz`
    * `rm boost_1_87_0.tar.gz`
    * `cd boost_1_87_0`
    * With admin rights and no other version of boost installed:
      * `./bootstrap.sh`
      * `./b2`
      * `sudo ./b2 install`
    * Otherwise:
      * `./bootstrap.sh --prefix=$BOOST_INSTALL_DIR`, where `$BOOST_INSTALL_DIR`
        is the full path to the directory where Boost should be installed.
      * `./b2 install`
      * Now, a local version of Boost is installed in the specified install
        directory. Later, while compiling an application using Boost (including
        the ISAAC server and the ISAAC examples), add
        `-DBoost_DIR=$BOOST_INSTALL_DIR`, where `$BOOST_INSTALL_DIR` is the path
        of the Boost install directory used above.

### Requirements for the in situ library and the examples using it

The ISAACConfig.cmake searches for these requirements. See
[`example/CMakeLists.txt`](./example/CMakeLists.txt) for an easy to adopt example.

* __alpaka__ 2.0.0+ for the abstraction of the acceleration device:
  * _From Source_:
    * __Boost__ should be installed before this, as alpaka may use it as a
      dependency.
    * alpaka supports multiple accelerators. If one wants to run ISAAC on GPU,
      __Cuda__ or __HIP__ should be installed beforehand and activated (see
      below).
    * `git clone https://github.com/alpaka-group/alpaka.git`
    * `cd alpaka`
    * `mkdir build && cd build`
    * `cmake .. -DCMAKE_INSTALL_PREFIX=$ALPAKA_INSTALL_DIR -Dalpaka_ACC_GPU_CUDA_ENABLE=ON`.
      The last option only must be included if acceleration on Nvidia GPUs is
      wanted. Go to the [alpaka manual](https://alpaka.readthedocs.io/en/stable/advanced/cmake.html#arguments)
      for infos on how to use other accelerators.
    * `cmake --install .`
    * Later, while compiling an application using alpaka (including the ISAAC
      examples), add `-Dalpaka_DIR=$ALPAKA_INSTALL_DIR`, where
      `$ALPAKA_INSTALL_DIR` is the path of the alpaka install directory used
      above.
* __IceT__ for combining the visualization created by the in situ plugin.
  * _Debian/Ubuntu_ (as part of Paraview):
    * `sudo apt-get install paraview-dev`
  * _From Source_:
    * `git clone https://gitlab.kitware.com/icet/icet.git`
    * `cd icet`
    * `mkdir build && cd build`
    * With admin rights and no other version of IceT installed:
      * `cmake ..`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `cmake .. -DCMAKE_INSTALL_PREFIX=$ICET_INSTALL_DIR`
      * `make install`
      * Now, a local version of IceT is installed in the specified install
        directory. Later, while compiling an application using ISAAC (including
        the examples), add `-DIceT_DIR=$ICET_INSTALL_DIR`, where
        `$ICET_INSTALL_DIR` is the path of the IceT install directory used above.
* __MPI__ for the communication on the cluster. This should be available on
  all clusters these days. However for a local testsystem OpenMPI is a commonly used
  version:
  * _Debian/Ubuntu_:
    * `sudo apt-get install libopenmpi-dev`
  * _From Source_:
    * `wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.gz`
    * `tar -xzf openmpi-5.0.10.tar.gz`
    * `rm openmpi-5.0.10.tar.gz`
    * `cd openmpi-5.0.10`
    * `mkdir build && cd build`
    * With admin rights and no other version of OpenMPI installed:
      * `../configure`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `../configure --prefix=$MPI_INSTALL_DIR`, where `$MPI_INSTALL_DIR` is
        the full (!) path to the directory where OpenMPI should be installed.
      * `make install`
      * Now, a local version of OpenMPI is installed in the specified install
        directory. Later while compiling an application using MPI (including the
        ISAAC examples) add `$MPI_INSTALL_DIR` to the CMake variable
        `CMAKE_MODULE_PATH` to use this version.
* __glm__ 1.0.0+ for the internal math types and functions
  * _From Source_:
    * `git clone https://github.com/g-truc/glm.git --depth 1 --branch 1.0.3`
    * `cd glm`
    * `mkdir build && cd build`
    * `cmake .. -DCMAKE_INSTALL_PREFIX=$GLM_INSTALL_DIR -DGLM_TEST_ENABLE=OFF`
    * `make install`
    * export `GLM_INSTALL_DIR` to your cmake prefix path (and add this to e.g.
      your profile)
      * `export CMAKE_PREFIX_PATH=$GLM_INSTALL_DIR:$CMAKE_PREFIX_PATH`

### Requirements for the server only

* __libwebsockets__ 2.1.1+ for the connection between server and an HTML5 client.
  * _From Source_:
    * `git clone https://github.com/warmcat/libwebsockets.git --depth 1 --branch v4.5-stable`
    * `cd libwebsockets`
    * `mkdir build && cd build`
    * With admin rights and no other version of libwebsockets installed:
      * `cmake ..`
        * This may fail if OpenSSL is not available. ISAAC itself does
          not support HTTPS connections at the moment anyway, thus it can be
          disabled with: `cmake -DLWS_WITH_SSL=OFF ..`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `cmake -DCMAKE_INSTALL_PREFIX=$LIBWEBSOCKETS_INSTALL_DIR ..`
        * This may fail if OpenSSL is not available. ISAAC itself does
          not support HTTPS connections at the moment anyway, thus it can be
          disabled with:
          `cmake -DLWS_WITH_SSL=OFF -DCMAKE_INSTALL_PREFIX=$LIBWEBSOCKETS_INSTALL_DIR ..`
      * `make install`
      * Now, a local version of libwebsockets is installed in the specified
        install directory. Later, while compiling the ISAAC server using
        libwebsockets, add
        `-DLibwebsockets_DIR=$LIBWEBSOCKETS_INSTALL_DIR/lib/cmake/libwebsockets`,
        where `$LIBWEBSOCKETS_INSTALL_DIR` is the path of the libwebsockets 
        install directory used above.
* __gStreamer__ is only needed, if streaming over RTP or the Twitch plugin shall
  be used. It should be possible to build gStreamer yourself, but it
  is strongly adviced - even from the gStreamer team themself - to use
  the prebuilt version of your distribution. The HML5 Client can show
  streams of a server without gStreamer. If gStreamer is not found, it is
  deactivated by default.
  * _Debian/Ubuntu_:
    * `sudo apt-get install libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 libgstreamer-plugins-good1.0-0 libgstreamer-plugins-bad1.0-0`

Building
--------

### Installing the library

To install the isaac library to use it in your project, go to directory `lib`
inside the isaac root folder, create a folder like `build`, and do the classic 
cmake magic:

* `git clone https://github.com/ComputationalRadiationPhysics/isaac.git`
* `cd isaac`
* `cd lib`
* `mkdir build && cd build`
* `cmake -DCMAKE_INSTALL_PREFIX=$ISAAC_LIB_DIR ..`
* (`sudo`) ` make install`

You don't need to call `make` before `make install` as the template library
does not need to be built until compiling the using application.

### The example

The building of the examples works similar, but the root directory of
the examples is the folder `example`, so after changing directory to
isaac (`cd isaac`) do:

* `cd example`
* `mkdir build && cd build`
* `cmake ..`
  * Don't forget the maybe needed `-DLIB_DIR=…` parameters
    needed for local installed libraries. E.g.
    `cmake -DIceT_DIR=$ICET_INSTALL_DIR/lib ..`
  * There are some options to (de)activate features of the library if they are
    not needed or not available on the system (like Cuda/HIP), which you can
    change with these lines before `..` (in `cmake ..`) or afterwards with
    `ccmake` or `cmake-gui`:
    * `-Dalpaka_ACC_GPU_CUDA_ENABLE=ON` or `-Dalpaka_ACC_GPU_HIP_ENABLE=ON` →
      Activates the CUDA / HIP accelerator in Alpaka. The used accelerator of
      ISAAC can be changed inside the file `example.cpp`, where at default CUDA
      is used as accelerator.
* `make install`

Afterwards you get the executable `isaac_example`.
To run this example, you need a running isaac server.

### The server

The server resides in the directory `server` and also uses CMake:

* `cd isaac`
* `cd server`
* `mkdir build && cd build`
* `cmake ..`
  * Don't forget the maybe needed `-DLIB_DIR=…` parameters
    needed for local installed libraries. E.g.
    `cmake -DLibwebsockets_DIR=$LIBWEBSOCKETS_INSTALL_DIR/lib/cmake/libwebsockets ..`
  * There are some options to (de)activate features of the server if they are
    not needed or not available on the system (like Gstreamer), which you can
    change with these lines before `..` (in `cmake ..`) or afterwards with
    `ccmake` or `cmake-gui`:
    * `-DISAAC_GST=OFF` → Deactivates GStreamer (Default if not found).
    * `-DISAAC_JPEG=OFF` → Deactivates JPEG compression. As already mentioned:
      This is not advised and will most probably leave ISAAC in an unusable
      state in the end.
    * `-DISAAC_SDL=ON` → Activates a plugin for showing the oldest not yet
      finished visualization in an extra window using `libSDL`. Of course, this
      option does not make much sense for most servers, as they don't have a
      screen or even an X server installed.
* `make`

If you want to install the server type

* (`sudo`) `make install`

Change the installation directory by adding

* `-DCMAKE_INSTALL_PREFIX=/your/path`

in the initial `cmake ..`

However, the server doesn't need to be installed and can also directly be called
from the build directory with

* `./isaac`

For more informations about parameters use `./isaac --help` or have a look in
the __[server documentation](https://computationalradiationphysics.github.io/isaac/doc/server/index.html)__.

### Testing

To test the server and an example, just start the server with `./isaac`, connect
to it with one of the HTML clients in the directory `client` (best is
`visualisation.html`) and start an example with `./isaac_example`. It should
connect to the server running on localhost and be observable and steerable. You
can run multiple instances of the example with `mpirun -c N ./isaac_example`
with the number of instances `N`. To exit the example, use the client or ctrl+C.

If the client and the isaac server are not located on the same system, it might
be required to create a tunnel between the two systems. In that case, you may
want to have a look into the __[tunnel guide](./TUNNEL.md)__.

### Versions

ISAAC has three different more or less independent (!) version strings.
The server and the library both have versions consisting of the

* major version number which is increased if the API compatibility is
  broken to older versions,
* minor version number, which indicates new features, but does not break
  code not using these new features, and
* patch version number, which is mostly for small bug fixes, but does not
  change much about the behaviour.

These version numbers are important if you want to use the ISAAC library
in your application or you want to extend the server with your own
meta data or image connector.

The last version string of isaac, the protocol version number, does only
consists of two version numbers: major and minor. The major protocol
version number must be the same for server and library. The minor version
number my differ, but in that case not all features of the protocol may
be used.


How to use in an own application
--------------------------------

For a deeper insight how to use ISAAC in a new application, have a look at the
__[library documentation](https://computationalradiationphysics.github.io/isaac/doc/library/index.html)__.
