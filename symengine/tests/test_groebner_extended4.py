"""
Source files referenced:
  - ComputationsBook/solving/chapter.m2 (C1)
"""

import pytest

from symengine import Symbol, groebner_basis, GroebnerBasis, Integer
from symengine.lib.symengine_wrapper import (
    normal_form,
    is_groebner,
    is_zero_dimensional,
    solve_poly_system,
)
from symengine.tests.groebner_oracle import contains_ideal, assert_correct_gb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_gb_exact(G, expected_polys, gens, order='degrevlex', modulus=0):
    """Assert computed GB equals expected set exactly and passes is_groebner.

    expected_polys must be a set of Basic expressions.  Coefficients in
    GF(p) expected polynomials must already be reduced mod p (0..p-1).
    """
    kw = {"order": order}
    if modulus != 0:
        kw["modulus"] = modulus
    assert is_groebner(list(G), gens, **kw), \
        f"is_groebner failed for {order}, modulus={modulus}"
    assert isinstance(expected_polys, set), \
        "expected_polys must be a set"
    actual_set = set(G)
    assert actual_set == expected_polys, \
        f"\nGB mismatch.\n  got:      {sorted(str(p) for p in actual_set)}\n  expected: {sorted(str(p) for p in expected_polys)}"


def _assert_ideal_membership(poly, G, gens, order='degrevlex', modulus=0):
    """Assert normal_form(poly, G, ...) == 0."""
    kw = {"order": order}
    if modulus != 0:
        kw["modulus"] = modulus
    nf = normal_form(poly, list(G), gens, **kw)
    assert nf == 0, \
        f"normal_form({poly}) = {nf}, expected 0"


# ============================================================================
# Phase A — Exact GF(101) Groebner bases
# ============================================================================

class TestGF101Groebner:
    """GF(101) exact Groebner basis tests ported from M2.

    GF(101) was chosen by M2 as a field large enough and prime.
    symengine requires a prime modulus > 1 (or 0 for QQ).
    """

    def test_m2_gf101_lex_linear(self):
        """engine/raw-gb.m2:150-154  -- linear system in GF(101), lex."""
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        gens = [x, y, z]
        F = [x*y - 1, x**2 - x, x**3 - z - 1]
        G = groebner_basis(F, x, y, z, order='lex', modulus=101)

        # M2 expected: {z, y-1, x-1}
        # In GF(101): -1 ≡ 100, so y-1 -> 100+y, x-1 -> 100+x.
        expected = {
            z,
            Integer(100) + y,
            Integer(100) + x,
        }
        _assert_gb_exact(G, expected, gens, order='lex', modulus=101)

    def test_m2_gf101_degrevlex_simple(self):
        """normal/testgb.m2:14-16  -- simple degrevlex basis in GF(101)."""
        a, b, c, d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')
        gens = [a, b, c, d]
        F = [a*b - c**2, b*c]
        G = groebner_basis(F, a, b, c, d, order='degrevlex', modulus=101)

        # M2 expected: {b*c, a*b - c**2, c**3}
        # In GF(101): a*b - c**2 -> a*b + 100*c**2
        expected = {
            b*c,
            a*b + Integer(100)*c**2,
            c**3,
        }
        _assert_gb_exact(G, expected, gens, order='degrevlex', modulus=101)

    def test_m2_gf101_degrevlex_twisted_cubic(self):
        """normal/testgb.m2:18-21  -- twisted cubic ideal in GF(101)."""
        a, b, c, d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')
        gens = [a, b, c, d]
        F = [a*b - c**2, a*c - d**2, b*c - c*d]
        G = groebner_basis(F, a, b, c, d, order='degrevlex', modulus=101)

        # M2 expected (7 elements).  All coefficients -1 ≡ 100 mod 101.
        expected = {
            b*c + Integer(100)*c*d,
            a*c + Integer(100)*d**2,
            a*b + Integer(100)*c**2,
            b*d**2 + Integer(100)*d**3,
            c**3 + Integer(100)*d**3,
            Integer(100)*a*d**3 + c**2*d**2,
            a**2*d**3 + Integer(100)*c*d**4,
        }
        _assert_gb_exact(G, expected, gens, order='degrevlex', modulus=101)

    def test_m2_gf101_redundant_generator(self):
        """engine/raw-gb.m2:66-75  -- GB with redundant generators."""
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        gens = [x, y, z]
        F = [x*y - y**2, x**2, y**4]
        G = groebner_basis(F, x, y, z, order='degrevlex', modulus=101)

        # M2 expected: {x*y - y**2, x**2, y**3}
        expected = {
            x*y + Integer(100)*y**2,
            x**2,
            y**3,
        }
        _assert_gb_exact(G, expected, gens, order='degrevlex', modulus=101)

    def test_m2_gf101_trivial_reduction(self):
        """engine/raw-gb.m2:38-44  -- trivial reduction with linear gens."""
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        gens = [x, y, z]
        F = [x, y, z**2, x*z + z**2]
        G = groebner_basis(F, x, y, z, order='degrevlex', modulus=101)

        # M2 expected: {y, x, z**2}  (no sign changes needed)
        expected = {x, y, z**2}
        _assert_gb_exact(G, expected, gens, order='degrevlex', modulus=101)


# ============================================================================
# Phase B — Integer / QQ coefficient Groebner bases
# ============================================================================

class TestIntegerGroebner:
    """Integer-coefficient (ZZ) and QQ Groebner basis tests ported from M2."""

    def test_m2_zz_linear(self):
        """normal/gbZZ-2.m2:3-9

        M2 computes over ZZ[x,y,z].  symengine uses QQ (field, modulus=0),
        so the GB is {1} (the ideal is the whole ring over QQ) whereas
        M2's ZZ GB is {19, z+3, y+3}.

        We verify that the QQ result is internally consistent: it's a
        Groebner basis that contains the inputs, and it's {1}.
        """
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        gens = [x, y, z]
        F = [-5*z + 4, 6*y - 1, 9*z + 8]
        G = groebner_basis(F, x, y, z, order='degrevlex')

        assert is_groebner(list(G), gens, order='degrevlex'), \
            "GB must be a Groebner basis"
        assert set(G) == {Integer(1)}, \
            f"over QQ the ideal is the whole ring, got {set(G)}"

    def test_m2_zz_univariate(self):
        """normal/gbZZ-2.m2:14-19

        M2 computes over ZZ[x] with expected GB {10, x}.
        Over QQ[x] the gcd is 1, so the GB is {1}.

        We verify the QQ result is internally consistent.
        """
        x = Symbol('x')
        gens = [x]
        F = [-6*x**3 + 10*x - 10, -2*x**2 - x, -x**3, -4*x**3 + 8*x**2]
        G = groebner_basis(F, x, order='degrevlex')

        assert is_groebner(list(G), gens, order='degrevlex'), \
            "GB must be a Groebner basis"
        assert set(G) == {Integer(1)}, \
            f"over QQ the ideal is the whole ring, got {set(G)}"

    def test_m2_qq_membership(self):
        """normal/gbZZbug.m2:3-11 (M2 uses QQ here despite the filename).

        f1 = x**2 + y**2 + z**2, f2 = x**2 + x*y + y**2 + x*z,
        f3 = x**2 + x*y + z**2 + y*z.
        I = (f1**3, f2**3, f3**3), g = f1**3 * f2**2 * f3.
        Assert g reduces to 0 modulo GB(I).  This is a pure ideal-membership
        check that exercises the Groebner engine with higher-degree inputs.
        """
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        gens = [x, y, z]
        f1 = x**2 + y**2 + z**2
        f2 = x**2 + x*y + y**2 + x*z
        f3 = x**2 + x*y + z**2 + y*z

        I = [f1**3, f2**3, f3**3]
        g = f1**3 * f2**2 * f3

        G = groebner_basis(I, x, y, z, order='degrevlex')
        _assert_ideal_membership(g, G, gens, order='degrevlex')


# ============================================================================
# Phase C — Elimination and solving
# ============================================================================

class TestElimination:
    """Elimination and lex-order tests ported from M2."""

    def test_m2_gf7_lex_saturate_consistency(self):
        """ComputationsBook/solving/chapter.m2:32-35

        The M2 test computes saturate(I, ideal(y)) which symengine does not
        expose.  Without saturation the direct GB of I is 1-dimensional
        (contains the y-axis) rather than zero-dimensional.  We verify the
        GB is internally consistent and document the difference.
        """
        y, x = Symbol('y'), Symbol('x')  # M2 ring: GF(7)[y,x]
        gens = [y, x]
        f1 = y**3*x**2 + 2*y**2*x + 3*x*y
        f2 = 3*y**2 + x*y - 3*y
        G = groebner_basis([f1, f2], y, x, order='lex', modulus=7)

        # With saturation, M2 expects y - 2*x - 1 and a univariate in x.
        # Without saturation, the ideal is 1-dimensional (non-zero-dim).
        assert is_groebner(list(G), gens, order='lex', modulus=7), \
            "GB must be a Groebner basis"
        assert len(list(G)) == 2, \
            f"expected 2 basis elements (unsatured), got {len(list(G))}"
        assert is_zero_dimensional(list(G), gens, order='lex', modulus=7) is False, \
            "ideal without saturation is 1-dimensional (contains the y-axis)"

    def test_m2_gf101_lex_high_degree(self):
        """engine/raw-gb.m2:168-178  -- high-degree lex basis in GF(101)."""
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        gens = [x, y, z]
        F = [x**2*y - y**3 - 1, x*y**2 - x - 1]
        G = groebner_basis(F, x, y, z, order='lex', modulus=101)

        # M2 expected:
        #   { y**7 - 2*y**5 + y**4 + y**3 - 2*y**2 - y + 1,
        #     x - y**6 + y**4 - y**3 + y + 1 }
        #
        # In GF(101): -2 ≡ 99, -1 ≡ 100 -> canonical form with 0..100 coeffs.
        expected = {
            Integer(1) + x + y + Integer(100)*y**3 + y**4 + Integer(100)*y**6,
            Integer(1) + Integer(100)*y + Integer(99)*y**2
            + y**3 + y**4 + Integer(99)*y**5 + y**7,
        }
        _assert_gb_exact(G, expected, gens, order='lex', modulus=101)

    def test_m2_elimination_lex_consistency(self):
        """Synthesised 0-dim lex elimination test.  x - y^2, y - z^2, z - 1."""
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        gens = [x, y, z]
        F = [x - y**2, y - z**2, z - 1]
        G = groebner_basis(F, x, y, z, order='lex')

        # Expected: {z - 1, y - 1, x - 1}
        expected = {z + Integer(-1), y + Integer(-1), x + Integer(-1)}
        _assert_gb_exact(G, expected, gens, order='lex')


# ============================================================================
# Phase D — FGLM and solve round-trips
# ============================================================================

class TestFGLM:
    """FGLM and solve round-trip tests ported from M2."""

    def test_m2_fglm_gf101_lex_to_degrevlex(self):
        """FGLM conversion between lex and degrevlex in GF(101).

        Uses the A1 ideal.  Both orders are tested: the ideal is
        zero-dimensional so FGLM applies.
        """
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        gens = [x, y, z]
        F = [x*y - 1, x**2 - x, x**3 - z - 1]

        G_lex = groebner_basis(F, x, y, z, order='lex', modulus=101)
        G_drl = groebner_basis(F, x, y, z, order='degrevlex', modulus=101)

        assert G_lex.is_zero_dimensional, "A1 ideal must be zero-dimensional"
        assert G_drl.is_zero_dimensional, "A1 ideal must be zero-dimensional"

        # FGLM lex -> degrevlex
        G_converted = G_lex.fglm('degrevlex')
        assert G_converted.order == 'degrevlex'
        assert is_groebner(list(G_converted), gens, order='degrevlex',
                           modulus=101), \
            "FGLM-converted basis must be a Groebner basis under degrevlex"

    def test_m2_solve_poly_system_gf101_unsupported(self):
        """solve_poly_system over GF(101) is not supported by symengine.

        The C++ solve backend only handles QQ; GF(p) roots are rejected
        with ValueError("solve_poly_system does not support roots over finite
        fields").  We verify this raises correctly.
        """
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        F = [x*y - 1, x**2 - x, x**3 - z - 1]
        with pytest.raises(ValueError, match="finite fields"):
            solve_poly_system(F, x, y, z, modulus=101)


# ============================================================================
# Cross-validations
# ============================================================================

class TestCrossValidation:
    """Additional cross-validation tests derived from M2 porting."""

    def test_m2_all_gf101_bases_are_groebner(self):
        """Every GF(101) basis from Phase A is indeed a Groebner basis."""
        cases = [
            (
                [Symbol('x'), Symbol('y'), Symbol('z')],
                [Symbol('x')*Symbol('y') - 1,
                 Symbol('x')**2 - Symbol('x'),
                 Symbol('x')**3 - Symbol('z') - 1],
                'lex',
                'A1',
            ),
            (
                [Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')],
                [Symbol('a')*Symbol('b') - Symbol('c')**2,
                 Symbol('b')*Symbol('c')],
                'degrevlex',
                'A2',
            ),
            (
                [Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')],
                [Symbol('a')*Symbol('b') - Symbol('c')**2,
                 Symbol('a')*Symbol('c') - Symbol('d')**2,
                 Symbol('b')*Symbol('c') - Symbol('c')*Symbol('d')],
                'degrevlex',
                'A3',
            ),
        ]
        for gens, F, order, label in cases:
            G = groebner_basis(F, *gens, order=order, modulus=101)
            assert is_groebner(list(G), gens, order=order, modulus=101), \
                f"{label}: {order} basis failed is_groebner"

    def test_m2_all_gf101_bases_contain_inputs(self):
        """Every GF(101) basis contains the input ideal."""
        cases = [
            (
                [Symbol('x'), Symbol('y'), Symbol('z')],
                [Symbol('x')*Symbol('y') - 1,
                 Symbol('x')**2 - Symbol('x'),
                 Symbol('x')**3 - Symbol('z') - 1],
                'lex',
                'A1',
            ),
            (
                [Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')],
                [Symbol('a')*Symbol('b') - Symbol('c')**2,
                 Symbol('b')*Symbol('c')],
                'degrevlex',
                'A2',
            ),
            (
                [Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')],
                [Symbol('a')*Symbol('b') - Symbol('c')**2,
                 Symbol('a')*Symbol('c') - Symbol('d')**2,
                 Symbol('b')*Symbol('c') - Symbol('c')*Symbol('d')],
                'degrevlex',
                'A3',
            ),
            (
                [Symbol('x'), Symbol('y'), Symbol('z')],
                [Symbol('x')*Symbol('y') - Symbol('y')**2,
                 Symbol('x')**2,
                 Symbol('y')**4],
                'degrevlex',
                'A4',
            ),
            (
                [Symbol('x'), Symbol('y'), Symbol('z')],
                [Symbol('x'), Symbol('y'),
                 Symbol('z')**2,
                 Symbol('x')*Symbol('z') + Symbol('z')**2],
                'degrevlex',
                'A5',
            ),
        ]
        for gens, F, order, label in cases:
            G = groebner_basis(F, *gens, order=order, modulus=101)
            assert contains_ideal(list(G), F, gens, order, modulus=101), \
                f"{label}: input not contained in computed GB"
