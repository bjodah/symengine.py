"""P2-5: output coefficient normalization over QQ (normalize= kwarg).

groebner_basis (and, via kwargs plumbing, GroebnerBasis.fglm) accept
normalize='monic' (default, unchanged: every polynomial scaled so its
leading coefficient is 1) or normalize='primitive' (SymPy/Singular-style:
each polynomial scaled to integer coefficients with gcd 1 and a positive
leading coefficient). Primitive only changes anything for the QQ coefficient
domain; GF(p) and symbolic coefficients accept the option and ignore it (see
GroebnerNormalization in symengine/polys/groebner.h).
"""

from fractions import Fraction
from math import gcd
from functools import reduce as functools_reduce

import pytest

from symengine import Symbol, groebner_basis, Integer

x, y = Symbol('x'), Symbol('y')


def _system_with_fractional_monic_output():
    # Chosen so the monic-normalized reduced basis has non-integer (1/2,
    # 1/3) leading-coefficient-normalized constant terms -- a real witness
    # that primitive normalization changes the emitted coefficients.
    return [2 * x**2 - 4 * y - 1, 3 * y**2 - 9 * x - 1]


def _int_coeffs(poly):
    d = poly.as_coefficients_dict()
    coeffs = []
    for v in d.values():
        assert isinstance(v, Integer) or int(v) == v
        coeffs.append(int(v))
    return coeffs


def test_normalize_default_is_monic():
    polys = _system_with_fractional_monic_output()
    G_default = groebner_basis(polys, x, y, order='degrevlex')
    G_monic = groebner_basis(polys, x, y, order='degrevlex', normalize='monic')
    assert G_default.polys == G_monic.polys


def test_normalize_primitive_has_integer_coefficients_with_gcd_one():
    polys = _system_with_fractional_monic_output()
    G_prim = groebner_basis(polys, x, y, order='degrevlex', normalize='primitive')
    assert len(G_prim.polys) > 0
    for p in G_prim.polys:
        coeffs = _int_coeffs(p)
        assert coeffs, f"expected nonzero polynomial, got {p}"
        g = functools_reduce(gcd, (abs(c) for c in coeffs))
        assert g == 1, f"{p} has coefficient gcd {g} != 1"


def test_normalize_primitive_matches_monic_up_to_positive_rational_rescale():
    polys = _system_with_fractional_monic_output()
    G_monic = groebner_basis(polys, x, y, order='degrevlex', normalize='monic')
    G_prim = groebner_basis(polys, x, y, order='degrevlex', normalize='primitive')
    assert len(G_monic.polys) == len(G_prim.polys)
    for pm, pp in zip(G_monic.polys, G_prim.polys):
        dm = pm.as_coefficients_dict()
        dp = pp.as_coefficients_dict()
        assert set(dm.keys()) == set(dp.keys())
        # Ratio primitive/monic must be the same positive rational for every
        # monomial (i.e. pp = k * pm for some k > 0).
        ratios = set()
        for mono, cm in dm.items():
            cm = Fraction(int(cm.p), int(cm.q)) if hasattr(cm, 'p') else Fraction(cm)
            cp = dp[mono]
            cp = Fraction(int(cp.p), int(cp.q)) if hasattr(cp, 'p') else Fraction(cp)
            ratios.add(cp / cm)
        assert len(ratios) == 1
        (k,) = ratios
        assert k > 0


def _monomial_degree(monomial):
    # monomial is 1, a bare generator, or a generator power (each term is
    # single-variable for the test system below).
    if monomial == 1:
        return 0
    if monomial.is_Pow:
        return int(monomial.args[1])
    return 1


def test_normalize_primitive_leading_coefficient_positive():
    polys = _system_with_fractional_monic_output()
    G_prim = groebner_basis(polys, x, y, order='degrevlex', normalize='primitive')
    for p in G_prim.polys:
        d = p.as_coefficients_dict()
        # The leading term (w.r.t. total degree, a proxy for the active
        # monomial order on these bivariate polys) must have a positive
        # coefficient under primitive normalization.
        leading_mono = max(d, key=_monomial_degree)
        assert d[leading_mono] > 0


def test_normalize_invalid_value_raises():
    polys = [x**2 - y]
    with pytest.raises(ValueError, match="normalize"):
        groebner_basis(polys, x, y, normalize='bogus')


def test_normalize_does_not_affect_gfp():
    polys = [x**2 - y, y**2 - x]
    G_monic = groebner_basis(polys, x, y, order='degrevlex', modulus=7,
                             normalize='monic')
    G_prim = groebner_basis(polys, x, y, order='degrevlex', modulus=7,
                            normalize='primitive')
    assert G_monic.polys == G_prim.polys


def test_fglm_accepts_normalize_kwarg():
    polys = _system_with_fractional_monic_output()
    G = groebner_basis(polys, x, y, order='degrevlex')
    G_lex_monic = G.fglm('lex', normalize='monic')
    G_lex_prim = G.fglm('lex', normalize='primitive')
    assert len(G_lex_monic.polys) == len(G_lex_prim.polys)
    for p in G_lex_prim.polys:
        coeffs = _int_coeffs(p)
        g = functools_reduce(gcd, (abs(c) for c in coeffs))
        assert g == 1
