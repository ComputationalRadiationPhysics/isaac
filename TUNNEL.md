![ISAAC](/isaac.png?raw=true "ISAAC")

ISAAC Tunnel Guide
==================

The idea behind ISAAC is to install the server on the login node, which
should be connectable both from the in-situ visualization running on
computation nodes as well as from the clients connecting from workstations
in the local network or over the internet.

However for different reasons this may not possible, e.g. if it is not
possible to install anything on the login node on yourself at all or
for security reasons as an open ISAAC server would accept connection
from anyone at the moment.

The solution is to run the ISAAC server on the head node of the cluster
instead of on the login node. However as most head nodes are not accessible
from the outside (on purpose) this needs tunneling for the clients, which
will be explained for ISAAC here.

Tunneling over SSH
==================

Regardless of whether Linux or Windows is used, in both cases ssh is used
for tunneling the connection. This has three main reasons:

* SSH is very easy to use and to install on both systems.
* It provides an user based security layer. Only users available on the
  login node are able to connect to the server.
* The whole comminunication is encrypted.

This explanation uses a different port than the original ISAAC port for
the html5 client (2459), to not collide with another ssh-session tunneling
isaac or even another isaac server instance running on the login node
itself. So in the html client the port needs to be changed. In this
explanation always port 2559 is used instead of port 2459.

Tunneling on Linux
==================

Most (if not all) Linux distributions should have the needed ssh client
already installed. All you need to do is to open a shell / terminal and
to launch ssh with these parameters:

* `ssh $USERNAME@$LOGIN_NODE -L 2559:$ISAAC_SERVER:2459`

This opens a shell on the login node, which will listen in the background
on port 2559 on the login node and forward any communication on it to `$ISAAC_SERVER`
on port 2459. `$USERNAME` is your username on the login node, `$LOGIN_NODE` your
system's login node and `$ISAAC_SERVER` the server on your local network, where
ISAAC runs. This may be your head node, but also a dedicated ISAAC server.
For the HZDR and it's hypnos cluster this would be for example:

* `ssh mustermann42@uts.fz-rossendorf.de -L 2559:hypnos5:2459`

After this the html client can connect to localhost on port 2559 (see the
left part of the second Windows screenshot), which will then be forwarded
to `$ISAAC_SERVER` over the `$LOGIN_NODE`.

Tunneling on Windows
====================

Unlike most Linux distributions Windows has not many useful programs
installed at default. So for the ssh tunneling an ssh client is needed.
I recommend the free software [putty](http://www.putty.org/) for this.

First you need to enter the url of the login server, called `$LOGIN_NODE`
in this screenshot:

![Putty screenshot 1](/documentation/tunnel_putty1.png?raw=true "Putty screenshot 1")

Afterwards you need to click Connection->SSH->Tunnels in the left menu
to get to this subwindow:

![Putty screenshot 2](/documentation/tunnel_putty2.png?raw=true "Putty screenshot 2")

On this screenshot you see also the client with the changed (!) port in the
background. Enter the source port and the destination as seen in this picture.
Of course `$ISAAC_SERVER` needs to be changed to the server running the
ISAAC server right now. Then click "Add" to enable the tunneling and afterwards
"Open" to connect to the login node. The window will close and a command
line prompt will appear asking for your username:

![Putty screenshot 3](/documentation/tunnel_putty3.png?raw=true "Putty screenshot 3")

Now enter here your username of the login node, press enter, enter the
password of your user and press enter again. Now you are connected to
the login node via ssh and if you connect (as seen in the second image)
to localhost on port 2559 your connection will be forwared to `$ISAAC_SERVER$`
on port 2459.

Finishing the tunneling
=======================

If you close the connection to login node (at best with Ctrl+D) the tunnel
will be closed, too. However if a tunnel connection is already established
ssh will not return (and the window will not close) until the last client
closes its connection to the ISAAC server.
