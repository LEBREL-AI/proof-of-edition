"""Proof of Edition, layer 3 extended: continuous watch of third-party inference APIs.

The audit in ``audit/`` re-executes a client's receipts against a reference. The watch
does the same job proactively and on a schedule, for APIs Lebrel does not operate:
it sends its own probe battery to every target (a lab's API, a host behind a router,
another router, a reference deployment Lebrel controls) and compares the answers
across targets and across time. It never sees a client's traffic; it only ever
compares Lebrel's own probes.

What it can and cannot say is written in ``watch/README.md``.
"""
