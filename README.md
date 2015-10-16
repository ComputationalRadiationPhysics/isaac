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
Simulation code has just to add some calls and settings to the insitu
template library. After that the server will notice when a simulation
is running and give the user some options to observe the computations
_on the fly_.

Requirements
------------

* libwebsockets for the connection between server and client
* Alpaka for the abstraction of the acceleration device
* IceT for combining the visualization created from the in situ plugin
* MPI for the communication of IceT

How to use
----------
To Do (In fact, it doesn't exist atm at all)

Licensing
---------

ISAAC is licensed under the GPLv3.

