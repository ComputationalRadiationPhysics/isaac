![ISAAC](/isaac.png?raw=true "ISAAC")

ISAAC Install Guide
=====================================================

Requirements
------------

Most dependencies are part of most distributions. However some need to
be built yourself nevertheless or the distribution versions are outdated.

### Requirements for the server and the in situ library

* __gcc__ / __g++__ for compiling the program at all. Version 4.8 __should__
  work, but 4.9 is recommended:
  * _Debian/Ubuntu_:
    * `sudo apt-get install gcc-4.9 g++-4.9 build-essential`
* __CMake__ for building everything:
  * _Debian/Ubuntu_:
    * `sudo apt-get install cmake cmake-curses-gui`
  * _From Source_ (As at least Version 3.1 is needed for the in-situ library):
    * `wget https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz`
    * `tar -zxvf cmake-3.5.2.tar.gz`
    * `rm cmake-3.5.2.tar.gz`
    * `cd cmake-3.5.2`
    * With admin rights and no other version of cmake installed:
      * `./configure`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `mkdir install`
      * `./configure --prefix=$CMAKE/install`, where where `$CMAKE` is
        the full (!) path of the cmake-3.5.2 directory.
      * `make install`
      * Now a local version of CMake is installed in the install directory in
        the cmake-3.5.2 folder. Later while compiling an application using
        CMake (including the ISAAC server and the ISAAC examples) use
        `$CMAKE/install/bin/cmake` instead of `cmake` and
        `$CMAKE/install/bin/ccmake` instead of `ccmake`, where `$CMAKE` is
        the path of the cmake-3.5.2 directory used above.
* __libjpeg__ or __libjpeg-turbo__ for (de)compressing the rendered image for the
  transmission:
  * _Debian/Ubuntu_:
    * `sudo apt-get install libjpeg-dev`
* __Jansson__ for the de- and encryption of the JSON messages transfered
  between server and client.
  * _Debian/Ubuntu_:
    * `sudo apt-get install libjansson-dev`
  * _From Source_:
    * `git clone https://github.com/akheron/jansson.git`
    * `cd jansson`
    * `mkdir build`
    * With admin rights and no other version of libjpeg installed:
      * `cd build`
      * `cmake ..`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `mkdir install`
      * `cd build`
      * `cmake .. -DCMAKE_INSTALL_PREFIX=../install`
      * `make install`
      * Now a local version of Jansson is installed in the install directory in
        the Jansson root folder. Later while compiling an application using
        Jansson (including the ISAAC server and the ISAAC examples) add
        `-DJansson_DIR=$JANSSON/install/lib/cmake/jansson`, where `$JANSSON` is
        the root folder of the Jansson source (the directory `git clone …` created).
* __Boost__ (at least 1.56) is needed, but only template libraries, so no
  system wide installation or static linking is needed here:
  * _Debian/Ubuntu_:
    * `sudo apt-get install libboost-dev`
  * _From Source_:
    * `wget http://sourceforge.net/projects/boost/files/boost/1.56.0/boost_1_56_0.tar.gz/download -O boost_1_56_0.tar.gz`
    * `tar -zxvf boost_1_56_0.tar.gz`
    * `rm boost_1_56_0.tar.gz`
    * `cd boost_1_56_0`
    * With admin rights and no other version of boost installed:
      * `./bootstrap.sh`
      * `./b2`
      * `sudo ./b2 install`
    * Otherwise:
      * `mkdir install`
      * `./bootstrap.sh --prefix=$BOOST/install`, where where `$BOOST` is
        the full (!) path of the boost_1_56_0 directory.
      * `./b2 install`
      * Now a local version of Boost is installed in the install directory in
        the boost_1_56_0 folder. Later while compiling an application using
        Boost (including the ISAAC server and the ISAAC examples) add
        `-DBoost_DIR=$BOOST/install`, where `$BOOST` is
        the path of the boost_1_56_0 directory.

### Requirements for the server only

* __libwebsockets__ for the connection between server and the HTML5 client.
  It is in steady development and the most recent version should be used:
  * _From Source_:
    * `git clone https://github.com/warmcat/libwebsockets.git`
    * `cd libwebsockets`
    * `mkdir build`
    * With admin rights and no other version of libwebsockets installed:
      * `cd build`
      * `cmake ..`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `mkdir install`
      * `cd build`
      * `cmake .. -DCMAKE_INSTALL_PREFIX=../install`
      * `make install`
      * Now a local version of libwebsockets is installed in the install
        directory in the libwebsockets root folder. Later while compiling the
        ISAAC server using libwebsockets add
        `-DLibwebsockets_DIR=$LIBWEBSOCKETS/install/lib/cmake/libwebsockets`, where
        `$LIBWEBSOCKETS` is the root folder of the libwebsockets source (the directory
        `git clone …` created).
* __gStreamer__ is only needed, if streaming over RTP or the Twitch plugin shall
  be used. It shall be possible to build gStreamer yourself, but it
  is strongly adviced - even from the gStreamer team themself - to use
  the prebuilt version of your distribution. The HML5 Client can show
  streams of a server without gStreamer.
  * _Debian/Ubuntu_:
    * `sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base0.10-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev`

### Requirements for the in situ library and the examples using it

The ISAACConfig.cmake searches for these requirements. See
example/CMakeLists.txt for an easy to adopt example.

* __Alpaka__ for the abstraction of the acceleration device. If only CUDA
  is used, this library is __not needed__:
  * _From Source_:
    * `git clone https://github.com/ComputationalRadiationPhysics/alpaka.git`
    * It is an header only library and doesn't need to be installed. However
      the root directory of the libary has to be added to the CMake Variable
      `CMAKE_MODULE_PATH`, e.g. with
      * `set(ALPAKA_ROOT "${CMAKE_SOURCE_DIR}/alpaka/" CACHE STRING  "The location of the alpaka library")`
      * `set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ALPAKA_ROOT}")`
* __CUDA__ for Nvidia accelerators. At least version 7.0 is needed for ISAAC, if
  CUDA acceleration is needed at all. If only OpenMP or TBB via Alpaka are needed,
  CUDA is not needed:
  * _Debian/Ubuntu_ (official repositories, at least Ubuntu 16.04 for CUDA 7.0):
    * `sudo apt-get install nvidia-cuda-dev`
  * _Debian/Ubuntu_ (directly from NVidia):
    * Download the most recent CUDA toolkit from here
      `https://developer.nvidia.com/cuda-downloads`. Choose `deb (network)`
      to download a package, which installes the NVidia Repository.
    * In the download folder of the package above do:
      * `sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb` (the name may differ,
        check what you downloaded)
      * `sudo apt-get update`
      * `sudo apt-get install cuda`
* __IceT__ for combining the visualization created from the in situ plugin.
  * _Debian/Ubuntu_ (as part of Paraview):
    * `sudo apt-get install paraview-dev`
  * _From Source_:
    * `git clone git://public.kitware.com/IceT.git`
    * `cd IceT`
    * `mkdir build`
    * With admin rights and no other version of IceT installed:
      * `cd build`
      * `cmake ..`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `mkdir install`
      * `cd build`
      * `cmake .. -DCMAKE_INSTALL_PREFIX=../install`
      * `make install`
      * Now a local version of IceT is installed in the install
        directory in the IceT root folder. Later while compiling
        an application using ISAAC (including the examples) add
        `-DIceT_DIR=$ICET/install/lib`, where
        `$ICET` is the root folder of IceT (the directory
        `git clone …` created).
* __MPI__ for the communication on the cluster. This should be available on
  all clusters these days. However for a local testsystem a common used
  version is OpenMPI:
  * _Debian/Ubuntu_:
    * `sudo apt-get install libopenmpi-dev`
  * _From Source_:
    * `git clone https://github.com/open-mpi/ompi.git`
    * `cd ompi`
    * `./autogen.pl`
    * With admin rights and no other version of OpenMPI installed:
      * `./configure`
      * `make`
      * `sudo make install`
    * Otherwise:
      * `mkdir install`
      * `./configure --prefix=$MPI/install`, where `$MPI` is
        the full (!) path of the openMPI directory.
      * `make install`
      * Now a local version of OpenMPI is installed in the install directory in
        the OpenMPI folder. Later while compiling an application using
        MPI (including the ISAAC examples) add `$MPI/install` to the
        CMake variable `CMAKE_MODULE_PATH` to use this version.

Building
--------

### The server

The server uses CMake. Best practice is to create a new directory (like
`build`) in the isaac root directory and change to it:
* `git clone https://github.com/ComputationalRadiationPhysics/isaac.git`
* `cd isaac`
* `mkdir build`
* `cd build`
* `cmake ..` (Don't forget the maybe needed `-DLIB_DIR=…` parameters
  needed for local installed libraries. E.g.
  `cmake -DLibwebsockets_DIR=$LIBWEBSOCKETS/install/lib/cmake/libwebsockets ..`)
* `make`

Installation of the server is not implemented yet, but you get
a single executable `isaac`, which can be run with `./isaac`.
For more informations about parameters use `./isaac --help` or have
a look in the __[server documentation](http://computationalradiationphysics.github.io/isaac/doc/server/index.html)__.

### The example

The building of the examples works similar, but the root directory of
the examples is the folder "example", so after changing directory to
isaac (`cd isaac`) do:

* `cd example`
* `mkdir build`
* `cd build`
* `cmake ..` (Don't forget the maybe needed `-DLIB_DIR=…` parameters
  needed for local installed libraries. E.g.
  `cmake -DIceT_DIR=$ICET/install/lib ..`)
* `make`

Afterwards you get executables `example_cuda`, `example_alpaka` or both.
Best practise is to use `ccmake ..` or `cmake-gui ..` to change the building options,
especially:

* `ISAAC_ALPAKA`, which enables using ALPAKA
* `ISAAC_CUDA`, which enables using CUDA

The used accelerator of Alpaka can be changed inside the file `example.cpp`.
At default OpenMP version 2 is used as Accelerator.

### Testing

To test the server and an example, just start the server with `./isaac`,
connect to it with one of the HTML clients in the directory `client` (best
is `interface_novlc.htm` with `Select stream (before observing!)`
set to `JPEG Stream (html5 only)`) and
start an example with `./example_cuda` or `./example_alpaka`. It should
connect to the server running on localhost and be observable and
steerable. You can run multiple instances of the example with
`mpirun -c N ./example_KIND` with the number of instances `N` and `KIND`
being `cuda` or `alpaka`. To exit the example use the client or Ctrl+C.


How to use in own application
-----------------------------

For a deeper insight how to use ISAAC in a new appliaction, have a look
at the [library documentation](http://computationalradiationphysics.github.io/isaac/doc/library/index.html).