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
    * With root rights and no other version of cmake installed:
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
    * With root rights and no other version of libjpeg installed:
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
        the root folder of the Jansson (the directory `git clone …` created).
* __Boost__ (at least 1.56) is needed, but only template libraries, so no
  system wide installation or static linking is needed here:
  * _Debian/Ubuntu_:
    * `sudo apt-get install libboost-dev`
  * _From Source_:
    * `wget http://sourceforge.net/projects/boost/files/boost/1.56.0/boost_1_56_0.tar.gz/download -O boost_1_56_0.tar.gz`
    * `tar -zxvf boost_1_56_0.tar.gz`
    * `rm boost_1_56_0.tar.gz`
    * `cd boost_1_56_0`
    * With root rights and no other version of boost installed:
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
    * With root rights and no other version of libwebsockets installed:
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
        `$LIBWEBSOCKETS` is the root folder of the libwebsockets (the directory
        `git clone …` created).
* __gStreamer__ is needed, if streaming over RTP or the Twitch plugin shall
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
    * As Alpaka is steady improved, the most recent version of Alpaka does not
      work with isaac. You need to force git to use a specific (working)
      version with:
      * `cd alpaka`
      * `git checkout e7b18db90cf4cf5fd6d6262adec9db176c3da8af`
    * It is an header only library and doesn't need to be installed. However
      the root directory of the libary has to be added to the CMake Variable
      `CMAKE_MODULE_PATH`, e.g. with
      * `set(ALPAKA_ROOT "${CMAKE_SOURCE_DIR}/alpaka/" CACHE STRING  "The location of the alpaka library")`
      * `set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ALPAKA_ROOT}")`
* __CUDA__ for Nvidia accelerators. At least version 7.0 is needed for ISAAC, if
  CUDA acceleration is needed at all. If only OpenMP and TBT is via Alpaka is needed,
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
    * With root rights and no other version of IceT installed:
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
        directory in the IceT root folder. Later while compiling the
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
    * With root rights and no other version of OpenMPI installed:
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
For more informations about parameters use `./isaac --help`.

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
Best practise is to use `ccmake ..` to change the building options,
especially:

* `ISAAC_ALPAKA`, which enables using ALPAKA
* `ISAAC_CUDA`, which enables using CUDA

The used accelerator of Alpaka can be changed inside the file `example.cu`.
At default OpenMP version 2 is used as Accelerator.

### Testing

To test the server and an example, just start the server with `./isaac`,
connect to it with one of the HTML clients in the directory `client` (best
is `interface_novlc.htm` with `Select stream (before observing!)`
set to `JPEG Stream (html5 only)`) and
start an example with `./example_cuda` or `./example_alpaka`. It should
connect to the server running on localhost and be observable and
steerable. You can run multiple instances of the example with
`mpirun -c N ./example_KIND` with the number of instances N and KIND
being cuda or alpaka. To exit the example use the client or Ctrl+C.


How to use in own application
-----------------------------

For using ISAAC in your own simulation first of all your need to include
the ISAAC library in your CMakeLists.txt as shown in the example:

* `find_package(ISAAC 0.1.0 REQUIRED)`.

To use the library you just need to include `isaac.hpp` with

* `#include <isaac.hpp>`

ISAAC is a template libary, which compiles supports for your sources
direct into its render kernel. This has the benefit, that the access of
your simulation data is high optimized. However you can't add sources
later, but add as many as you want and deactivate them if not needed.

For every source an own class needs to be defined. This class has to
implement this following things:
* `static const size_t feature_dim` containing the feature dimension of
  the source. For a vector field this would be e.g. most probably 3, for
  a scalar field 1.
* `static const bool has_guard` determines, whether the source has a
  guard for access of data outside the later defined local dimensions.
* `static const bool persistent` determines, whether ths source is
  persistent or needs to be copied after calling the update function.
* speaking of it. Every source class needs an update function
  `ISAAC_HOST_INLINE void update(bool enabled)`. The definition has to
  look exactly like this. This function is called right before the
  rendering and before update is called for the next source the data
  will be copied for non persistent sources. `enabled` tells you,
  whether the source will be drawn at all. It is up to you to not update
  arrays in that case. Non persistent sources will be accessed at all in
  that case.
* `ISAAC_HOST_INLINE static std::string getName()` has to return the
  name of the source.
* an overload of the [] operator, which looks like this:
  ```
  ISAAC_NO_HOST_DEVICE_WARNING
  ISAAC_HOST_DEVICE_INLINE isaac_float_dim< feature_dim >
  operator[] (const isaac_int3 nIndex) const
  {
    ...
  }
  ```
  This function has to return the value at the position nIndex in the
  isaac data format isaac_float_dim<N>, which is an struct like this
  ```
  template <int N>
  struct
  {
     isaac_floatN value;
  };
  ```
  where value has the members .x, .y, .z and/or .w depending the feature
  dimension of your source.

Everything else in the class is up to you, but keep in mind, it may be
copied from time to time, so be careful with memory in constructors and
destructors.

For a working example of such classes have a look in example/example.cu.
If you are using CUDA, you can ignore all ISAAC_ALPAKA ifdefs and only
look at the CUDA else branch. I will continue explaining using the
CUDA example code. If you are using alpaka you should understand the
alpaka specific differences yourself.

Now you need a `boost::fusion::list` with all your classes like
```
TestSource1 testSource1;
TestSource2 testSource2;

using SourceList = boost::fusion::list
<
    TestSource1,
    TestSource2
>;

SourceList sources( testSource1, testSource2 );
```
`sources` is needed for the initialization of isaac, which is happening
now with the creation of an instance of `IsaacVisualization`.
```
auto visualization = new IsaacVisualization <
    SimDim,
    SourceList,
    std::vector<size_t>,
    1024,
    std::vector<float>,
    isaac::DefaultController,
    isaac::DefaultCompositor
> (
    name,
    0,
    server,
    port,
    framebuffer_size,
    global_size,
    local_size,
    position,
    sources,
    scaling
);
```
with the template parameters
* `SimDim` is an integral type containing the dimension of the
  simulation, e.g. boost::mpl::int_<3>
* `SourceList` is the type we defined earlier containing a boost fusion
  list of source classes
* `std::vector<size_t>` is the type used for storing the dimension of
  the global and local volume and the position of the second in the
  first.
* `1024` is the size of the transfer function used for every source
* `std::vector<float>` is a user defined type of scaling
* `isaac::DefaultController` is a trait for setting up the projection,
  may be changed for stereoscopy.
* `isaac::DefaultCompositor` is a trait for combining or postprocessing
  the rendered image(s), may be changed for stereoscopy.

and the contructor parameters

* `name`: The name of the vizualisation
* `0`: The rank, which will communicate with the isaac server
* `server`: the url of the server
* `port`: the port of the server
* `framebuffer_size`: A vector containing the width and height of the
  rendered image
* `global_size`: The size of the whole volume
* `local_size`: The size of the local subvolume
* `position`: The position of the local subvolume in the global volume
* `sources`: The just created runtime instance of the SourceList.
* `scaling`: An optional scaling of the simulation data.

Now it is possible to define and describe some metadata the client shall
see. It doesn't need to be defined at this point, but every here defined
datum will be shown in the list of available simulations.
`visualization->getJsonMetaRoot()` returns an instance of `json_t*`, a
json root, where you can add more members. See the example and the
jansson documention for more details.

After defining the metadata we can now connect to the isaac server with
`visualization->init()`. If 0 is returned the connection is established.

Everytime you want to send a rendered view of your data (e.g. after
ever time step of your simulation) call
`visualization->doVisualization(META_MASTER);`
`META_MASTER` means, that only the master rank (in most cases rank 0)
will send metadata. If you choose `META_MERGE` the metadata of every
rank is merged via MPI before. To add meta data just use
`visualization->getJsonMetaRoot()` again before calling the
`doVisualization` function. After this function call the returned json
root from `getJsonMetaRoot` is empty again.

`doVisualization` itself returns a `json_t*` root, too. Every data put
into "metadata" from the client will be forwarded to _every_ application
rank. You _have_ to call json_decref to the result of this function!
Even if you don't want to use the metadata, call it at least this way:
```
json_decref( visualization->doVisualization( META_MASTER ) );
```
Otherwise you will open a memory leak.

At the end of the simulation just delete the visualization object. The
connection will close and the simulation will disappear for every client
of the isaac server.

