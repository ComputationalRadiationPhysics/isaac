ISAAC - In Situ Animation of Accelerated Computations
=====================================================

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
* `mkdir build`
* `cd build`
* `cmake ..`

to create the Makefiles. Missing dependencies and variables will be
reported. If everything is setup, just call
* `make`

to build isaac (and the optional example)
Now you can just start ISAAC with
* `./isaac`

The Paraneter --help will give you some options like the used ports to
setup up. The default port for the client is 2459 (the year when the
Serenity laid keel) and for the insitu connections 2560.

If you build the examples you can also start one or multiple instances
with
* `./example_cuda`

or
* `./example_alpaka`

or with more than one binary
* `mpirun -c COUNT ./example_cuda`

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

In the setup code of your simulation you need to create an instance of
IsaacVisualization:

* `IsaacVisualization* myVisualization = new IsaacVisualization( … );`

TODO: Finish this explanation. :P

Licensing
---------

ISAAC is licensed under the LGPLv3.

