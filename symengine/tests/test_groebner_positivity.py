"""Tests for the positivity-aware assumption classifier.

Report 25 SS3.3(5): if a user guarantees sign conditions on parameters, many
genericity assumptions become mechanically dischargeable, but mixed-sign
combinations (e.g. a determinant C2*C3 - C4*C1) never are. These tests cover
the structural classify_assumptions() rules directly, plus the
GroebnerBasis/PolySolveResult delegating methods and an end-to-end scenario.
"""

import pytest

from symengine import (
    Symbol, Integer, groebner_basis, classify_assumptions,
    solve_poly_system_ex,
)


C1, C2, C3, C4 = Symbol('C1'), Symbol('C2'), Symbol('C3'), Symbol('C4')


def test_bare_symbol_positive_satisfied():
    r = classify_assumptions([C1], positive=(C1,))
    assert r == {'satisfied': [C1], 'undecidable': []}


def test_bare_symbol_negative_satisfied():
    r = classify_assumptions([C1], negative=(C1,))
    assert r == {'satisfied': [C1], 'undecidable': []}


def test_bare_symbol_nonzero_satisfied():
    r = classify_assumptions([C1], nonzero=(C1,))
    assert r == {'satisfied': [C1], 'undecidable': []}


def test_bare_symbol_no_knowledge_undecidable():
    r = classify_assumptions([C1])
    assert r == {'satisfied': [], 'undecidable': [C1]}


def test_nonzero_number_satisfied():
    r = classify_assumptions([Integer(5)])
    assert r == {'satisfied': [Integer(5)], 'undecidable': []}
    r = classify_assumptions([Integer(-3)])
    assert r == {'satisfied': [Integer(-3)], 'undecidable': []}


def test_zero_number_undecidable():
    # Never claim 'satisfied' for a value that provably is not nonzero.
    r = classify_assumptions([Integer(0)])
    assert r == {'satisfied': [], 'undecidable': [Integer(0)]}


def test_mul_both_positive_satisfied():
    r = classify_assumptions([C1 * C2], positive=(C1, C2))
    assert r == {'satisfied': [C1 * C2], 'undecidable': []}


def test_mul_one_unknown_undecidable():
    r = classify_assumptions([C1 * C2], positive=(C1,))
    assert r == {'satisfied': [], 'undecidable': [C1 * C2]}


def test_add_both_positive_satisfied():
    r = classify_assumptions([C1 + C2], positive=(C1, C2))
    assert r == {'satisfied': [C1 + C2], 'undecidable': []}


def test_add_both_negative_satisfied():
    r = classify_assumptions([C1 + C2], negative=(C1, C2))
    assert r == {'satisfied': [C1 + C2], 'undecidable': []}


def test_pow_nonzero_exponent_satisfied():
    r = classify_assumptions([C1**2], nonzero=(C1,))
    assert r == {'satisfied': [C1**2], 'undecidable': []}


def test_pow_positive_base_satisfied():
    r = classify_assumptions([C1**3], positive=(C1,))
    assert r == {'satisfied': [C1**3], 'undecidable': []}


def test_determinant_mixed_sign_undecidable():
    """The report's canonical example: even with every symbol known
    positive, C2*C3 - C4*C1 is a mixed-sign combination and must never be
    reported 'satisfied'."""
    assumption = C2 * C3 - C4 * C1
    r = classify_assumptions([assumption],
                             positive=(C1, C2, C3, C4))
    assert r == {'satisfied': [], 'undecidable': [assumption]}


def test_positive_minus_negative_satisfied():
    """C1 - C2 with C1 positive, C2 negative: this is positive minus a
    negative, i.e. a sum of two positives structurally, so it is
    satisfied."""
    r = classify_assumptions([C1 - C2], positive=(C1,), negative=(C2,))
    assert r == {'satisfied': [C1 - C2], 'undecidable': []}


def test_empty_knowledge_everything_undecidable_except_numbers():
    assumptions = [C1, C1 * C2, C1 + C2, Integer(7)]
    r = classify_assumptions(assumptions)
    assert r['satisfied'] == [Integer(7)]
    assert set(r['undecidable']) == {C1, C1 * C2, C1 + C2}


def test_groebner_basis_classify_assumptions_delegates():
    x, y = Symbol('x'), Symbol('y')
    G = groebner_basis([C1 * x**2 - y, x + y], x, y, order='degrevlex',
                       algorithm='buchberger')
    assumptions = G.stats['genericity_assumptions']
    assert len(assumptions) > 0
    r = G.classify_assumptions(nonzero=(C1,))
    assert set(r['satisfied']) | set(r['undecidable']) == set(assumptions)
    # C1 itself (a bare symbol pivot) must be discharged by nonzero=(C1,).
    assert C1 in r['satisfied']


def test_polysolveresult_classify_assumptions_delegates():
    x, y = Symbol('x'), Symbol('y')
    result = solve_poly_system_ex([C1 * x - 1, y - x], x, y)
    r = result.classify_assumptions(nonzero=(C1,))
    assert set(r['satisfied']) | set(r['undecidable']) \
        == set(result.assumptions)


def test_end_to_end_determinant_lands_in_undecidable():
    """groebner_basis on a 2x2 fully symbolic linear system records the
    Cramer's-rule determinant as a genericity assumption; even when every
    parameter is known positive, that determinant must be classified
    'undecidable', not 'satisfied'."""
    x, y = Symbol('x'), Symbol('y')
    G = groebner_basis([C1 * x + C2 * y - 1, C3 * x + C4 * y - 1],
                       x, y, order='lex')
    assumptions = G.stats['genericity_assumptions']
    determinant_candidates = [a for a in assumptions
                              if {C1, C2, C3, C4}.issubset(a.free_symbols)]
    assert determinant_candidates, \
        "expected a determinant-like assumption involving all 4 parameters"

    r = G.classify_assumptions(positive=(C1, C2, C3, C4))
    for det in determinant_candidates:
        assert det in r['undecidable']
        assert det not in r['satisfied']
