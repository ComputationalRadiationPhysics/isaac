![ISAAC](/isaac.png?raw=true "ISAAC")

In Situ Animation of Accelerated Computations
=====================================================

![Wakefield visualization from PIConGPU](/example_renderings/picongpu_wakefield_1.png?raw=true "Wakefield visualization from PIConGPU")

About ISAAC
-----------

These days, many computation-intensive codes, like physics or biology simulations, 
run on GPU-accelerated hardware, which is itself distributed in big compute
clusters, communicating via MPI with each other. 
The goal of ISAAC is to visualize this data without the need to copy it
to the host, while using the high computation speed of these accelerators.

ISAAC consists of two parts: The server and the in-situ library.
Furthermore, a client which is able to show the transferred stream and meta data
is needed. A reference HTML5 client is provided here,
but needs to be adapted to specific simulations and is not part of ISAAC
itself (but still hosted in the repository).

Simulation code just has to add some calls and settings to the in-situ
template library. After that, the server will notice when a simulation
is running and give the user some options to observe the computations
_on the fly_. It is also possible to send meta data back to the
simulation, e.g., to restart it with improved settings.

Installing requirements, building and using in own application
--------------------------------------------------------------

Please see in [INSTALL.md](./INSTALL.md) for installing, building and
using ISAAC and its dependencies.
If you need to install ISAAC on a server not accessible from the outside,
you need to [tunnel the connections](./TUNNEL.md) of the clients.
A more detailed __documentation__ about using ISAAC __can be
[found here](https://computationalradiationphysics.github.io/isaac)__ (_currently
outdated_).

Client
------

Inside the client directory, the HTML client `visualisation.html` can be found.

It lets you connect to your visualisations directly from your web browser and
allows to change settings and steer the visualisation. For visualisations of
__[PIConGPU](https://github.com/ComputationalRadiationPhysics/picongpu)__
with ISAAC enabled, it will also display meta data like the cell count or 
current time step.

As an alternative, a Python client for Jupyter Notebooks is also in developement 
and can be found __[here](https://github.com/ComputationalRadiationPhysics/isaac-python)__.

Licensing
---------

ISAAC is licensed under the LGPLv3.

