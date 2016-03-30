![ISAAC](/isaac.png?raw=true "ISAAC")

In Situ Animation of Accelerated Computations
=====================================================

![Wakefield visualization from PIConGPU](/examples/picongpu_wakefield_1.png?raw=true "Wakefield visualization from PIConGPU")

About ISAAC
-----------

Many computations like physics or biologists simulations these days
run on accelerated hardware like CUDA GPUs or Intel Xeon Phi, which are
itself distributed in a big compute cluster communicating over MPI. The
goal of ISAAC is to visualize this data without the need to download it
to the host while using the high computation speed of the accelerator.

ISAAC insists of two parts: The server and the insitu library.
Furthermore there needs to be a client, which is able to show the
transfered stream and meta data. An example HTML5 client is provided,
but needs to be adapted to specific simulations and is not part of ISAAC
itself.

Simulation code has just to add some calls and settings to the insitu
template library. After that the server will notice when a simulation
is running and give the user some options to observe the computations
_on the fly_. It is also possible to send meta data back to the
simulation, e.g. to restart it with improved settings.

Requirements
------------

Most dependencies are part of most distributions. However some need to
be built yourself nevertheless or the distribution versions are outdated.
Every link given in this chapter is a link to a git repository and needs
to be cloned with `git clone $LINK`, where $LINK is a link like
`https://github.com/ComputationalRadiationPhysics/isaac.git`.

### Requirements for the server and the in situ library

* __libjpeg__ or __libjpeg-turbo__ for (de)compressing the rendered image for the
  transmission. It should be part of most distributions, but can also
  be found here: `git@github.com:libjpeg-turbo/libjpeg-turbo.git`.
* __Jansson__ for the de- and encryption of the JSON messages transfered
  between server and client. The library is part of most distributions,
  but can also be found here: `https://github.com/akheron/jansson.git`.
* __CMake__ for building everything.

### Requirements for the server only

* __libwebsockets__ for the connection between server and the HTML5 client.
  It is in steady development and the most recent version (which should be
  used) can be found here: `https://github.com/warmcat/libwebsockets.git`.
* __gStreamer__ is needed, if streaming over RTP or the Twitch plugin shall
  be used. It shall be possible to build gStreamer yourself, but it
  is strongly adviced - even from the gStreamer team themself - to use
  the prebuilt version of your distribution. The HML5 Client can show
  streams of a server without gStreamer.
  
### Requirements for the in situ library and the examples using it

The ISAACConfig.cmake searches for these requirements. See
example/CMakeLists.txt for an easy to adopt example.

* __Alpaka__ for the abstraction of the acceleration device. The library can
  be found here: `https://github.com/ComputationalRadiationPhysics/alpaka.git`.
  It is an header only library and doesn't need to be installed. However
  the root directory of the libary has to be added to the CMake Variable
  `CMAKE_MODULE_PATH`, e.g. with
  ```set(ALPAKA_ROOT "${CMAKE_SOURCE_DIR}/alpaka/" CACHE STRING  "The location of the alpaka library")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ALPAKA_ROOT}")```
  If only CUDA is used, this library is not needed.
* __CUDA__ for Nvidia accelerators. The most recent CUDA toolkit can be
  downloaded at Nvidia's webpage: `https://developer.nvidia.com/cuda-downloads`.
  If only ALPAKA without CUDA is used, this toolkit is not needed.
* __IceT__ for combining the visualization created from the in situ plugin.
  In Ubuntu IceT is part of the ParaView package `paraview-dev`, but can
  also be found here: `git://public.kitware.com/IceT.git`.
* __MPI__ for the communication on the cluster. This should be available on
  all clusters these days. However for a local testsystem a common used
  version is OpenMPI, which is found in all distributions, but can also
  be compiled from source from this repository:
  `https://github.com/open-mpi/ompi.git`
* __Boost__ (at least 1.56) is needed, but only template libraries, so no
  installation or static linking is needed here. If the version of the
  distribution is too old, it can be found here:
  `https://github.com/boostorg/boost.git`. Unlike most (read: all) other
  projects, boost doesn't use make or cmake, but an own build mechanism.
  It's all build around the script `bootstrap.sh`.
  `./bootstrap.sh --help` will give you some help for building and
  installing the library.

Known issues
------------

* If streaming over twitch or another rtmp compatible service is used,
  but the rtmp port (1935) ist blocked or a wrong url passed, the server
  will crash because of the underlying gStreamer rtmp implementation.

Building
--------

### The server

The server uses CMake. Best practice is to create a new directory (like
`build`) in the isaac root directory and change to it:
```mkdir build
cd build```
With `cmake ..` (plain cli), `ccmake ..` (with ncurses gui) or
`cmake-gui ..` (qt-gui) the server can be built. If a library is missing
cmake will tell you and you should be able to set the lookup directories
for the missing libraries. If everything is fine you need to generate
the Makefile (only needed for ccmake and cmake-gui) and call it with
`make`. Installation of the server is not implemented yet, but you get
a single executable `isaac`, which can be run with `./isaac`.
For mor informations about parameters use `./isaac --help`.

### The example

The building of the examples works similar, but the root directory of
the examples is the folder "example", so
```cd example
mkdir build
cd build```
will prepare everything for building it with cmake. The rest is exactly
as above (but with other requirements). Depending on the cmake-flags
`ISAAC_CUDA` and `ISAAC_ALPAKA` the files `example_cuda` and/or
`example_alpaka` are generated.

### Testing

To test the server and an example, just start the server with `./isaac`,
connect to it with on of the HTML clients in the directory `client` and
start an example with `example_cuda` or `example_alpaka`. It should
connect to the server running on localhost and be observable and
steerable. You can run multiple instances of the example with
`mpirun -c N ./example_KIND` with the number of instances N and KIND
being cuda or alpaka. To exit the example use the client or Ctrl+C.


How to use in own application
-----------------------------

For using ISAAC in your own simulation first of all your need to include
the ISAAC library in your CMakeLists.txt as shown in the examples:
```find_package(ISAAC 0.1.0 REQUIRED)```.

To include it you just need to include isaac.hpp.
ISAAC is a template libary, which compiles supports for your sources
direct into its render kernel. This has the benefit, that the access of
your simulation data is high optimized. However you can't add sources
later, but add as many as you want and deactivate them if not needed.

For every source an own class needs to be defined. This class has to
implement this following things:
* `static const std::string name` containing the name of the source
* `static const size_t feature_dim` containing the feature dimension of
  the source. For a vector field this would be e.g. most probably 3, for
  a scalar field 1.
* `static const bool has_guard` determines, whether the source has a
  guard for access of data outside the later defined local dimensions
* `static const bool persistent` determines, whether ths source is
  persistent or needs to be copied after calling the update function.
* speaking of it. Every source class needs an update function
  `ISAAC_HOST_INLINE void update(bool enabled)`. The definition has to
  look exactly like this. This function is called right before the
  rendering and before update is called for the next source the data
  will be copied for non persistent sources. `enabled` tells you,
  whether the source will be drawn at all. It is up to you to not update
  arrays in that case. Non persistent sources will be accessed at all in
  that case
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

For a working example of such classes have a look in insitu/example.cu.
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
    1024
> (
    name,
    0,
    server,
    port,
    framebuffer_size,
    global_size,
    local_size,
    position,
    sources
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

Licensing
---------

ISAAC is licensed under the LGPLv3.

