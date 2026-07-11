"""Tests for the coefficient-complexity resource limit (max_coefficient_ops).

This guards against coefficient swell in the symbolic (Expression) coefficient
domain -- the practical failure mode for parametric systems with many
symbolic parameters (see docs/reports/26-BENCHMARK-CAMPAIGN.md). The C++ side
implements it in ComputationCheckpoint::check_coefficient_complexity,
enforced from normal_form_impl and gated behind should_poll() (it is only
cheap to poll, not to check on every single reduction step). Because of that
polling gate, tripping it reliably from Python needs a finer poll cadence
than the (unchanged) default of 1024 -- hence cancellation_check_interval is
exercised alongside max_coefficient_ops below.
"""

import pytest

from symengine import Symbol, groebner_basis, symbols


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
