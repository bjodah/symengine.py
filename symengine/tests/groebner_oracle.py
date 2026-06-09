"""Shared correctness oracle for Groebner basis testing.

Every test in the extended testing campaign routes through this module.
The key insight (from docs/reports/09-11): is_groebner(G) only checks that G
is a GB of *its own* ideal <G>, NOT that <G> == <input>.  The functions here
check all three correctness properties:

  (1) <G> = <F>  (same ideal -- both inclusions)
  (2) G is a Groebner basis
  (3) The reduced Groebner basis is unique for a fixed (ideal, order, domain)
"""

import multiprocessing
import subprocess
import sys
import os

from symengine.lib.symengine_wrapper import (
    normal_form,
    is_groebner as _is_groebner,
    is_reduced_basis as _is_reduced_basis,
)


# ---------------------------------------------------------------------------
# Core oracle functions
# ---------------------------------------------------------------------------

def contains_ideal(G, F, gens, order, modulus=0):
    """True iff <F> is contained in <G>: every polynomial in F reduces to 0
    modulo G.  This is the check that catches 'dropped generators' (e.g. the
    m4gb bug)."""
    G, F = list(G), list(F)
    kw = {"order": order}
    if modulus:
        kw["modulus"] = modulus
    return all(normal_form(p, G, gens, **kw) == 0 for p in F)


def same_ideal(A, B, gens, order, modulus=0):
    """True iff <A> == <B>: every element of A reduces to 0 mod B AND vice
    versa.  Generalizes the original _same_ideal from test_groebner.py."""
    return (contains_ideal(B, A, gens, order, modulus) and
            contains_ideal(A, B, gens, order, modulus))


def assert_correct_gb(F, gens, G, order, modulus=0, golden_size=None,
                      reference=None):
    """The full correctness contract for a computed Groebner basis G of input F.

    Checks:
      (a) is_groebner(G)                              -- property (2)
      (b) contains_ideal(G, F)                        -- <F> subset <G>
      (c) if reference is given: same_ideal(G, reference) -- property (1)
      (d) if golden_size is not None: len(G) == golden_size
      (e) if reference is given and both are reduced: set equality -- property (3)

    Raises AssertionError on any violation.
    """
    G_list = list(G)
    F_list = list(F)
    kw = {"order": order}
    if modulus:
        kw["modulus"] = modulus

    # (a) G must be a Groebner basis of its own ideal
    assert _is_groebner(G_list, gens, **kw), \
        f"is_groebner failed: G is not a Groebner basis (algo={getattr(G, 'algorithm', '?')})"

    # (b) Every input polynomial must reduce to 0 modulo G
    #     (catches "dropped generators" -- the m4gb bug)
    assert contains_ideal(G_list, F_list, gens, order, modulus), \
        "contains_ideal failed: some input polynomial does not reduce to 0 mod G (<F> not subset <G>)"

    # (c) Reference agreement: G must generate the same ideal as the reference
    if reference is not None:
        ref_list = list(reference)
        assert same_ideal(G_list, ref_list, gens, order, modulus), \
            "same_ideal failed: G and reference do not generate the same ideal"
        # (e) If both are reduced, the sets must be identical (uniqueness)
        if _is_reduced_basis(G_list, gens, **kw) and \
           _is_reduced_basis(ref_list, gens, **kw):
            assert set(G_list) == set(ref_list), \
                "reduced-basis uniqueness violated: reduced G != reduced reference"

    # (d) Golden size check
    if golden_size is not None:
        assert len(G) == golden_size, \
            f"golden size mismatch: expected {golden_size}, got {len(G)}"


# ---------------------------------------------------------------------------
# Subprocess wrapper (for hanging/segfaulting computations)
# ---------------------------------------------------------------------------

_SENTINEL_TIMEOUT = "TIMEOUT"
_SENTINEL_ERROR = "ERROR"


def _worker(fn, args, kwargs, queue):
    """Target for a subprocess-safe computation."""
    try:
        result = fn(*args, **kwargs)
        queue.put(("OK", result))
    except Exception as e:
        queue.put(("ERROR", e))


def run_with_timeout(fn, *args, timeout=30, **kwargs):
    """Run fn(*args, **kwargs) in a subprocess with a wall-clock timeout.

    Returns the result on success, or a sentinel string "TIMEOUT" / "ERROR"
    on failure.  This prevents a hanging/segfaulting C++ engine from blocking
    the test suite.

    Note: fn and its arguments must be picklable for multiprocessing.
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(fn, args, kwargs, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return _SENTINEL_TIMEOUT
    if not q.empty():
        status, value = q.get()
        if status == "OK":
            return value
        return _SENTINEL_ERROR
    return _SENTINEL_ERROR


# ---------------------------------------------------------------------------
# Algorithm enumeration
# ---------------------------------------------------------------------------

def all_algorithms(modulus=0):
    """Return the list of algorithms to test for a given domain.

    For QQ: buchberger, f5b, mogvw (m4gb is QQ-only via a different path).
    For GF(p): buchberger, f5b, mogvw, m4gb.
    'auto' is tested separately since it's a routing layer.
    """
    algos = ["buchberger", "f5b", "mogvw"]
    if modulus:
        algos.append("m4gb")
    return algos


def auto_algorithm():
    """Return 'auto' for tests that specifically check auto-routing."""
    return "auto"
