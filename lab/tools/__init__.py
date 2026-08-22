"""Development tools for Resonance Lab.

Includes mock_daemon, a wire-protocol stand-in for the real daemon, used to test
the client without a GPU. It reproduces the daemon's awkward behaviours on
purpose: silent out-of-range ignores, sticky injection parameters, stress only
on request.
"""

