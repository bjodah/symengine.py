"""Tests for the coefficient-complexity resource limit (max_coefficient_ops)
and (P2-2) the canonical limits=/cancellation_token= control surface.

The max_coefficient_ops-specific tests guard against coefficient swell in the
symbolic (Expression) coefficient domain -- the practical failure mode for
parametric systems with many symbolic parameters (see
docs/reports/26-BENCHMARK-CAMPAIGN.md). The C++ side implements it in
ComputationCheckpoint::check_coefficient_complexity, enforced from
normal_form_impl and gated behind should_poll() (it is only cheap to poll,
not to check on every single reduction step). Because of that polling gate,
tripping it reliably from Python needs a finer poll cadence than the
(unchanged) default of 1024 -- hence cancellation_check_interval is exercised
alongside max_coefficient_ops below.

The P2-2 section below covers the canonical (non-legacy) control mechanism:
limits={...} maps onto SymEngine::ComputationLimits and cancellation_token=
wraps SymEngine::GroebnerCancellationToken (an AtomicCancellationToken).
Routing semantics: limits= installs an explicit GroebnerOptions.control,
which fully supersedes the legacy numeric kwargs (max_s_pairs,
max_reduction_steps, max_milliseconds, ... -- see GroebnerEntryScope in
groebner.cpp). A cancellation_token= WITHOUT limits= deliberately goes
through the legacy cancellation_token field instead, so it composes with
the legacy numeric kwargs rather than silently disabling them.
"""

import threading
import time

import pytest

from symengine import Symbol, groebner_basis, symbols, CancellationToken
from symengine.tests.groebner_corpus_extended import EXTENDED_SYSTEMS_QQ


def _symbolic_3x3_linear_system():
    """9 matrix parameters (a11..a33) + 3 RHS parameters (b1..b3) = 12
    parameters total, over unknowns x, y, z. Mirrors
    benchmarks/groebner_bench_parametric.cpp's "param_3x3_linear_*" system:
    its eliminated pivots genuinely swell in the symbolic coefficient
    domain, making it a real (not synthetic) witness for this limit."""
    a11, a12, a13, a21, a22, a23, a31, a32, a33, b1, b2, b3 = symbols(
        "a11 a12 a13 a21 a22 a23 a31 a32 a33 b1 b2 b3")
    x, y, z = symbols("x y z")
    polys = [
        a11 * x + a12 * y + a13 * z - b1,
        a21 * x + a22 * y + a23 * z - b2,
        a31 * x + a32 * y + a33 * z - b3,
    ]
    return polys, [x, y, z]


def test_max_coefficient_ops_default_is_unlimited():
    polys, gens = _symbolic_3x3_linear_system()
    G = groebner_basis(polys, *gens, order="lex")
    assert len(G.polys) == 3


def test_max_coefficient_ops_tiny_limit_trips():
    polys, gens = _symbolic_3x3_linear_system()
    with pytest.raises(RuntimeError, match="resource limits"):
        groebner_basis(polys, *gens, order="lex",
                       max_coefficient_ops=8,
                       cancellation_check_interval=1)


def test_max_coefficient_ops_zero_is_a_no_op():
    polys, gens = _symbolic_3x3_linear_system()
    G = groebner_basis(polys, *gens, order="lex",
                       max_coefficient_ops=0,
                       cancellation_check_interval=1)
    assert len(G.polys) == 3


def test_max_coefficient_ops_does_not_affect_qq():
    """The metric only applies to the symbolic (Expression) coefficient
    domain; QQ systems must be unaffected even by a tiny limit."""
    x, y = Symbol("x"), Symbol("y")
    polys = [x**2 - y, x**3 - x]
    G = groebner_basis(polys, x, y, order="lex",
                       max_coefficient_ops=1,
                       cancellation_check_interval=1)
    assert len(G.polys) > 0


def test_max_coefficient_ops_does_not_affect_gfp():
    """The metric only applies to the symbolic (Expression) coefficient
    domain; GF(p) systems must be unaffected even by a tiny limit."""
    x, y = Symbol("x"), Symbol("y")
    polys = [x**2 - y, x**3 - x]
    G = groebner_basis(polys, x, y, order="lex", modulus=32003,
                       max_coefficient_ops=1,
                       cancellation_check_interval=1)
    assert len(G.polys) > 0


# ===========================================================================
# P2-2a -- limits={...} (canonical ComputationLimits, mirrors the cwrapper's
# CComputationLimits setters one-to-one by field name).
# ===========================================================================


def _nontrivial_qq_basis():
    x, y = Symbol("x"), Symbol("y")
    return [x**2 - y, y**2 - x, x * y - 1], [x, y]


def test_limits_max_reduction_steps_trips():
    polys, gens = _nontrivial_qq_basis()
    with pytest.raises(RuntimeError, match="resource limits"):
        groebner_basis(polys, *gens, limits={"max_reduction_steps": 1})


def test_limits_generous_succeeds():
    polys, gens = _nontrivial_qq_basis()
    G = groebner_basis(polys, *gens,
                       limits={"max_reduction_steps": 10_000,
                               "max_basis_size": 10_000,
                               "max_spairs_processed": 10_000})
    assert len(G.polys) > 0


@pytest.mark.parametrize("reserved_key", [
    "max_spairs_generated", "max_basis_total_terms", "max_polynomial_terms",
])
def test_limits_reserved_key_raises(reserved_key):
    polys, gens = _nontrivial_qq_basis()
    with pytest.raises(ValueError, match="reserved"):
        groebner_basis(polys, *gens, limits={reserved_key: 1})


def test_limits_unknown_key_raises():
    polys, gens = _nontrivial_qq_basis()
    with pytest.raises(ValueError, match="unknown limits key"):
        groebner_basis(polys, *gens, limits={"max_reducton_steps": 1})


def test_limits_must_be_a_dict():
    polys, gens = _nontrivial_qq_basis()
    with pytest.raises(TypeError, match="limits must be a dict"):
        groebner_basis(polys, *gens, limits="not a dict")


def test_limits_none_is_a_no_op():
    polys, gens = _nontrivial_qq_basis()
    G = groebner_basis(polys, *gens, limits=None)
    assert len(G.polys) > 0


def test_limits_supersedes_legacy_fields():
    """When limits= is given, it takes over control entirely -- a legacy
    max_reduction_steps alongside a generous limits= must not trip on the
    legacy value (GroebnerEntryScope zeroes the legacy fields whenever
    options.control is set)."""
    polys, gens = _nontrivial_qq_basis()
    G = groebner_basis(polys, *gens,
                       max_reduction_steps=1,
                       limits={"max_reduction_steps": 10_000})
    assert len(G.polys) > 0


def test_limits_max_basis_size_trips():
    polys, gens = _nontrivial_qq_basis()
    with pytest.raises(RuntimeError, match="resource limits"):
        groebner_basis(polys, *gens, limits={"max_basis_size": 1})


def test_cancellation_token_composes_with_legacy_limits():
    """A token WITHOUT limits= must not disable the legacy numeric kwargs:
    it is routed through the legacy cancellation_token field (not an
    explicit control, which would supersede them), so a tiny legacy
    max_reduction_steps alongside an (uncancelled) token still trips."""
    polys, gens = _nontrivial_qq_basis()
    tok = CancellationToken()
    with pytest.raises(RuntimeError, match="resource limits"):
        groebner_basis(polys, *gens,
                       cancellation_token=tok,
                       max_reduction_steps=1)
    # ... and the token itself still works on the same call shape.
    tok.cancel()
    with pytest.raises(RuntimeError, match="cancel"):
        groebner_basis(polys, *gens,
                       cancellation_token=tok,
                       max_reduction_steps=1_000_000)


# ===========================================================================
# P2-2b -- cancellation_token= (GroebnerCancellationToken / CancellationToken)
# ===========================================================================


def test_cancellation_token_pre_cancelled_raises():
    polys, gens = _nontrivial_qq_basis()
    tok = CancellationToken()
    tok.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        groebner_basis(polys, *gens, cancellation_token=tok)


def test_cancellation_token_not_cancelled_succeeds():
    polys, gens = _nontrivial_qq_basis()
    tok = CancellationToken()
    assert tok.is_cancelled() is False
    G = groebner_basis(polys, *gens, cancellation_token=tok)
    assert len(G.polys) > 0


def test_cancellation_token_reset_allows_reuse():
    polys, gens = _nontrivial_qq_basis()
    tok = CancellationToken()
    tok.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        groebner_basis(polys, *gens, cancellation_token=tok)
    tok.reset()
    assert tok.is_cancelled() is False
    G = groebner_basis(polys, *gens, cancellation_token=tok)
    assert len(G.polys) > 0


def test_cancellation_token_rejects_wrong_type():
    polys, gens = _nontrivial_qq_basis()
    with pytest.raises(TypeError, match="CancellationToken"):
        groebner_basis(polys, *gens, cancellation_token="not a token")


def _slow_qq_system():
    """eco6 (msolve/eco6-qq.ms) under lex order takes ~9-10s wall time with
    this build's default algorithm selection -- long enough that a 0.5s
    sleep-then-cancel from another thread interrupts it mid-computation
    (not before it starts, not after it finishes), while still keeping this
    test's total wall time in the sub-second range. See
    groebner_corpus_extended.EXTENDED_SYSTEMS_QQ['eco6'].
    """
    sysdef = EXTENDED_SYSTEMS_QQ["eco6"]
    gens = [Symbol(g) for g in sysdef["gens"]]
    env = {g.name: g for g in gens}
    polys = [eval(p, {}, env) for p in sysdef["polys"]]
    return polys, gens


def test_cancellation_token_cancels_in_flight_computation():
    """Cancel a computation from a different thread while it is running.

    The GIL is released during groebner_basis's nogil block, so a token
    created before the call and cancelled from another thread mid-flight
    must interrupt it well before the (generous, safety-net-only)
    max_milliseconds budget -- if cancellation is broken, this test fails on
    the wall-time assertion (or times out entirely) instead of hanging
    forever, because max_milliseconds=30000 is passed as a backstop. When
    cancellation works, wall time is dominated by the 0.5s sleep, not the
    30s safety budget or the ~9.7s uncancelled runtime.
    """
    polys, gens = _slow_qq_system()
    tok = CancellationToken()
    result = {}

    def run():
        try:
            result["basis"] = groebner_basis(
                polys, *gens, order="lex",
                cancellation_token=tok,
                max_milliseconds=30_000)
        except BaseException as exc:  # noqa: BLE001 - propagate to main thread
            result["error"] = exc

    thread = threading.Thread(target=run)
    t0 = time.time()
    thread.start()
    time.sleep(0.5)
    tok.cancel()
    thread.join(timeout=35)
    elapsed = time.time() - t0

    assert not thread.is_alive(), "computation did not stop after cancel()"
    assert "error" in result, f"expected cancellation error, got {result}"
    assert isinstance(result["error"], RuntimeError)
    assert "cancel" in str(result["error"]).lower()
    # Should stop shortly after the 0.5s sleep + cancel(), nowhere near the
    # 30s safety net or the ~9.7s uncancelled runtime.
    assert elapsed < 5.0, f"cancellation took too long: {elapsed:.2f}s"


def test_cancellation_token_reset_then_fresh_run_succeeds():
    """After a cancel + reset, the token works for a fresh successful run."""
    polys, gens = _nontrivial_qq_basis()
    tok = CancellationToken()
    tok.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        groebner_basis(polys, *gens, cancellation_token=tok)
    tok.reset()
    G = groebner_basis(polys, *gens, cancellation_token=tok, order="lex")
    assert len(G.polys) > 0
