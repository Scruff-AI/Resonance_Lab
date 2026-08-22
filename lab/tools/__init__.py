"""Standalone tools that support the lab but are not part of the server.

    python -m lab.tools.mock_daemon

speaks the real wire protocol on the real ports with no GPU, so the client can
be exercised and changed without borrowing time on a real run.
"""
