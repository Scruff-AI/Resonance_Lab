"""Resonance Lab — a browser client for a running Khra'gixx daemon.

Nothing in this package changes the physics. It reads the daemon's existing ZMQ
streams, sends only commands the daemon already accepted, and renders views that
strip the forcing out so structure underneath it can be seen.

Run the server with:

    python -m lab.server

and, if there is no GPU to hand, a stand-in daemon with:

    python -m lab.tools.mock_daemon

See lab/README.md for the rest.
"""

__version__ = "0.2.0"
