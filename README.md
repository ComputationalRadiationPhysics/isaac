ISAAC - In Situ Animation of Accelerated Computations
=====================================================

![Wakefield visualization from PIConGPU](/examples/picongpu_wakefield.png?raw=true "Wakefield visualization from PIConGPU")

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

Most of these requirements are git submodules and can be found in
requirement-submodules.

* libwebsockets for the connection between server and the example client
* Jansson for the de- and encryption of the JSON messages transfered
  between server and client
* Alpaka for the abstraction of the acceleration device
* IceT for combining the visualization created from the in situ plugin
* MPI for the communication of IceT
* Boost (at least 1.5.6) is needed, but only template libraries, so no
  installation or static linking is needed here

What does work
--------------

ISAAC is in very active production now, so some features are already
working, but some not.
This does work:
* Registering a simulation at the control server
* Unregisteringa simulation from the control server
* Showing all available simulations
* Registering to observe such a simulation
* Unregistering from Observing
* For observing clients:
    * Getting json meta data from the simulation
    * Sending json meta data to the simulation
* For registered simulations
    * Sending json meta data per frame
    * Sending an image per frame
* Creating a renderer image (raycast or iso surface) from the data on
  the acceleration device at the compute nodes
* Streaming a compressed video stream to the client. The stream can be
  embedded in the json meta data or alternative RTP Streams can be
  created.
* Setting up the visualization parameters like
    * Maximal functor chain lengths
    * The functor chain itself
    * Transfer functions
    * Whether to use interpolation or not
* Precompiling of all kernelvariants for more speed

How to use
----------

TODO: Writing FindISAAC.cmake or similar.

ISAAC uses CMake. So just do:
```
mkdir build
cd build
cmake ..
```

to create the Makefiles. Missing dependencies and variables will be
reported. If everything is setup, just call `make`.

to build isaac (and the optional example)
Now you can just start ISAAC with `./isaac`

The Paraneter --help will give you some options like the used ports to
setup up. The default port for the client is 2459 (the year when the
Serenity laid keel) and for the insitu connections 2560.

If you build the examples you can also start one or multiple instances
with `./example_cuda` or `./example_alpaka` or with more than one binary
with `mpirun -c COUNT ./example_cuda`.

If you open the interface.htm in the subfolder client you should be
able to connect to ISAAC, to see the running example and observe it. For
existing the example or isaac just press ^C (Ctrl + C).

For using ISAAC in your own simulation first of all your need to include
the isaac.h from the subfolder insitu and. For now it is easist to just
copy and paste this header library. Make sure, that this file finds the
folder "isaac" with sub include files. Keep in mind, that you need
* jansson
* alpaka
* IceT and
* MPI
to use this library. A CMakeFile to check for this is in work.

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

