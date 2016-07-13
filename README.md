![ISAAC](/isaac.png?raw=true "ISAAC")

In Situ Animation of Accelerated Computations
=====================================================

![Wakefield visualization from PIConGPU](/example_renderings/picongpu_wakefield_1.png?raw=true "Wakefield visualization from PIConGPU")

About ISAAC
-----------

Many computations like physics or biologists simulations these days
run on accelerated hardware like CUDA GPUs or Intel Xeon Phi, which are
itself distributed in a big compute cluster communicating over MPI. The
goal of ISAAC is to visualize this data without the need to download it
to the host while using the high computation speed of the accelerator.

ISAAC insists of two parts: The server and the insitu library.
Furthermore there needs to be a client, which is able to show the
transfered stream and meta data. A reference HTML5 client is provided,
but needs to be adapted to specific simulations and is not part of ISAAC
itself (but still in the repository).

Simulation code has just to add some calls and settings to the insitu
template library. After that the server will notice when a simulation
is running and give the user some options to observe the computations
_on the fly_. It is also possible to send meta data back to the
simulation, e.g. to restart it with improved settings.

Installing requirements, building and using in own application
--------------------------------------------------------------

Please see in [INSTALL.md](./INSTALL.md) for installing, building and
using ISAAC.
If you need to install ISAAC on a server not accessible from the outside
you need to [tunnel the connections](./TUNNEL.md) of the clients.
A more detailed __documentation__ about using ISAAC __can be
[found here](http://computationalradiationphysics.github.io/isaac)__.

Client
------

Inside the client directory lay three files:
* interface.htm
* interface_vlc.htm
* interface_presentation.htm
The very first uses direct JSON injection, but no real streaming protocol
like RTP. The second can use RTP, but needs the (free) vlc browser plugin
installed to work. Furthermore the server most be able to directly send
UDP packages to you, which are blocked by most firewalls. Last but not
least this adds some latency for the h264 encoding. However the stream itself
will use way less bandwidth. The last client version is for presentation
purposes, the table with meta data values is not shown, but some are shown
directly in the stream box. E.g. if you visualize
__[PIConGPU](https://github.com/ComputationalRadiationPhysics/picongpu)__
with ISAAC enabled this will show you the count of particles and cells
used.

Known issues
------------

* If streaming over twitch or another rtmp compatible service is used,
  but the rtmp port (1935) ist blocked or a wrong url passed, the server
  will crash because of the underlying gStreamer rtmp implementation.

Licensing
---------

ISAAC is licensed under the LGPLv3.

