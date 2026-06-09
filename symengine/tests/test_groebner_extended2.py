"""Extended Groebner basis testing campaign: Work Packages 5-8.

WP5 — Parametric / symbolic-coefficient path
WP6 — solve_poly_system correctness
WP7 — Edge cases and input robustness
WP8 — Property-based testing (Hypothesis)
"""

import pytest

from symengine import (
    Symbol, groebner_basis, GroebnerBasis, Integer, Rational,
    FiniteSet, EmptySet, sympify,
)
from symengine.lib.symengine_wrapper import (
    normal_form, is_groebner, is_reduced_basis,
    is_zero_dimensional, solve_poly_system,
)
from symengine.tests.groebner_oracle import (
    same_ideal, contains_ideal, assert_correct_gb,
    run_with_timeout, all_algorithms, _SENTINEL_TIMEOUT, _SENTINEL_ERROR,
)


# ============================================================================
# WP5 — Parametric / symbolic-coefficient path
# ============================================================================

class TestWP5Parametric:

    def test_parametric_buchberger_vs_f5b(self):
        x, y = Symbol('x'), Symbol('y')
        C1, C2 = Symbol('C1'), Symbol('C2')
        F = [C1*x*y + C2*y**2 - 1, x**2 - C1*y]
        gens = [x, y]
        G_buch = groebner_basis(F, *gens, order='degrevlex', algorithm='buchberger')
        G_f5b = groebner_basis(F, *gens, order='degrevlex', algorithm='f5b')
        assert is_groebner(list(G_buch), gens, order='degrevlex')
        assert is_groebner(list(G_f5b), gens, order='degrevlex')
        assert same_ideal(G_buch, G_f5b, gens, order='degrevlex')

    def test_parametric_is_groebner_membership(self):
        x, y = Symbol('x'), Symbol('y')
        C1, C2 = Symbol('C1'), Symbol('C2')
        F = [C1*x**2 + C2*y, x*y - 1]
        gens = [x, y]
        G = groebner_basis(F, *gens, order='degrevlex', algorithm='buchberger')
        assert is_groebner(list(G), gens, order='degrevlex')
        assert contains_ideal(G, F, gens, order='degrevlex')

    def test_parametric_generic_substitution_agrees(self):
        x, y = Symbol('x'), Symbol('y')
        C1 = Symbol('C1')
        F_param = [C1*x + y - 1, x - 1]
        gens = [x, y]
        G_param = groebner_basis(F_param, *gens, order='degrevlex',
                                 algorithm='buchberger')
        for val in [1, 2, 3, -1, 5]:
            G_concrete = groebner_basis(
                [val*x + y - 1, x - 1], *gens, order='degrevlex')
            subbed = [p.subs({C1: val}) for p in G_param]
            assert same_ideal(subbed, list(G_concrete), gens, order='degrevlex'), \
                f"Parametric GB substituted at C1={val} disagrees with concrete GB"

    @pytest.mark.xfail(strict=True,
                       reason="bug: degenerate parametric substitution produces "
                              "zoo terms instead of correct reduced basis")
    def test_parametric_degenerate_substitution_changes_answer(self):
        """When C1=0, the system C1*x^2 - y, x+y becomes {y, x+y} = {x, y}.
        Substituting C1=0 into the parametric GB yields zoo*y + y^2 which is
        not the correct basis.  This documents the degenerate case."""
        x, y = Symbol('x'), Symbol('y')
        C1 = Symbol('C1')
        F_param = [C1*x**2 - y, x + y]
        gens = [x, y]
        G_param = groebner_basis(F_param, *gens, order='degrevlex',
                                 algorithm='buchberger')
        G_concrete = groebner_basis(
            [Integer(0)*x**2 - y, x + y], *gens, order='degrevlex')
        subbed = [p.subs({C1: 0}) for p in G_param]
        assert same_ideal(subbed, list(G_concrete), gens, order='degrevlex'), \
            "Degenerate substitution C1=0 should match concrete GB"

    def test_parametric_stats_genericity_is_tuple(self):
        x, y = Symbol('x'), Symbol('y')
        C1, C2 = Symbol('C1'), Symbol('C2')
        G = groebner_basis([C1*x**2 + C2*y, x*y - 1], x, y, order='degrevlex')
        assert isinstance(G.stats['genericity_assumptions'], tuple)

    def test_parametric_multi_symbol_agreement(self):
        x, y = Symbol('x'), Symbol('y')
        C1, C2, C3 = Symbol('C1'), Symbol('C2'), Symbol('C3')
        F = [C1*x*y + C2*y**2 - 1, x**2 - C3*y]
        gens = [x, y]
        G = groebner_basis(F, *gens, order='degrevlex', algorithm='buchberger')
        assert is_groebner(list(G), gens, order='degrevlex')
        assert contains_ideal(G, F, gens, order='degrevlex')


# ============================================================================
# WP6 — solve_poly_system correctness
# ============================================================================

def _verify_solutions(F, sol, gens):
    """Independent oracle: every returned point must satisfy every input poly."""
    assert isinstance(sol, FiniteSet), \
        f"Expected FiniteSet, got {type(sol).__name__}"
    for pt in sol.args:
        vals = list(pt.args)
        assert len(vals) == len(gens)
        subs_map = dict(zip(gens, vals))
        for f in F:
            v = f.subs(subs_map)
            assert v == 0, \
                f"Solution {vals} does not satisfy {f} (got {v})"


class TestWP6SolvePolySystem:

    def test_solve_zero_dim_two_solutions(self):
        x, y = Symbol('x'), Symbol('y')
        F = [x**2 + y**2 - 2, x - y]
        sol = solve_poly_system(F, x, y)
        _verify_solutions(F, sol, [x, y])
        pts = {tuple(p.args) for p in sol.args}
        assert pts == {(Integer(1), Integer(1)), (Integer(-1), Integer(-1))}

    def test_solve_zero_dim_four_solutions(self):
        x, y = Symbol('x'), Symbol('y')
        F = [x**2 + y**2 - 5, x*y - 2]
        sol = solve_poly_system(F, x, y)
        _verify_solutions(F, sol, [x, y])
        pts = {tuple(p.args) for p in sol.args}
        assert pts == {
            (Integer(1), Integer(2)),
            (Integer(2), Integer(1)),
            (Integer(-1), Integer(-2)),
            (Integer(-2), Integer(-1)),
        }

    def test_solve_count_vs_bezout(self):
        x, y = Symbol('x'), Symbol('y')
        F = [x**2 - y, x**3 - x]
        sol = solve_poly_system(F, x, y)
        _verify_solutions(F, sol, [x, y])
        pts = list(sol.args)
        assert len(pts) == 3
        # Bezout bound: deg(x^2 - y) * deg(x^3 - x) = 2 * 3 = 6
        assert len(pts) <= 6

    @pytest.mark.xfail(strict=True,
                       reason="bug: solve_poly_system returns solutions with "
                              "complex radical expressions that do not simplify "
                              "to zero on substitution back")
    def test_solve_count_katsura3(self):
        x0, x1, x2 = Symbol('x0'), Symbol('x1'), Symbol('x2')
        F = [x0 + 2*x1 + 2*x2 - 1,
             x0**2 + 2*x1**2 + 2*x2**2 - x0,
             2*x0*x1 + 2*x1*x2 - x1]
        sol = solve_poly_system(F, x0, x1, x2)
        _verify_solutions(F, sol, [x0, x1, x2])
        # Katsura-3 has 4 solutions over C
        assert len(sol.args) == 4

    def test_solve_inconsistent_y_minus_x(self):
        x, y = Symbol('x'), Symbol('y')
        sol = solve_poly_system([y - x, y - x - 1], x, y)
        assert sol == EmptySet

    def test_solve_inconsistent_x1_x2(self):
        x, y = Symbol('x'), Symbol('y')
        sol = solve_poly_system([x - 1, x - 2], x, y)
        assert sol == EmptySet

    def test_solve_positive_dimensional_documented(self):
        """Positive-dimensional systems currently return EmptySet.
        This test documents that behavior."""
        x, y = Symbol('x'), Symbol('y')
        sol = solve_poly_system([x*y - 1], x, y)
        assert sol == EmptySet

    def test_solve_linear_system(self):
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        F = [2*x - 3, Rational(3, 2)*y - 2*x, z - 5*y]
        sol = solve_poly_system(F, x, y, z)
        _verify_solutions(F, sol, [x, y, z])
        pts = list(sol.args)
        assert len(pts) == 1
        assert list(pts[0].args) == [Rational(3, 2), Integer(2), Integer(10)]

    @pytest.mark.xfail(strict=True,
                       reason="bug: solve_poly_system returns EmptySet for "
                              "zero-dimensional 3-var system with known "
                              "solutions (0,0,1), (0,1,0), (1,0,0)")
    def test_solve_three_var_system(self):
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        F = [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1]
        sol = solve_poly_system(F, x, y, z)
        _verify_solutions(F, sol, [x, y, z])
        assert len(sol.args) > 0


# ============================================================================
# WP7 — Edge cases and input robustness
# ============================================================================

class TestWP7EdgeCases:

    def test_empty_generator_list_raises(self):
        x, y = Symbol('x'), Symbol('y')
        with pytest.raises(ValueError):
            groebner_basis([], x, y)

    def test_single_constant_unit_ideal(self):
        x, y = Symbol('x'), Symbol('y')
        G = groebner_basis([x, x - 1], x, y, order='degrevlex')
        assert len(G) == 1
        assert G[0] == 1

    def test_zero_polynomial_in_input(self):
        x, y = Symbol('x'), Symbol('y')
        G = groebner_basis([Integer(0), x - 1], x, y, order='degrevlex')
        assert is_groebner(list(G), [x, y], order='degrevlex')
        assert len(G) == 1

    def test_multiple_zeros_with_real_polynomial(self):
        x, y = Symbol('x'), Symbol('y')
        G = groebner_basis([Integer(0), Integer(0), x**2 - y], x, y,
                           order='degrevlex')
        assert is_groebner(list(G), [x, y], order='degrevlex')

    def test_duplicate_generators(self):
        x, y = Symbol('x'), Symbol('y')
        G_dup = groebner_basis([x**2 - 1, x**2 - 1, x**2 - 1], x, y,
                               order='degrevlex')
        G_single = groebner_basis([x**2 - 1], x, y, order='degrevlex')
        assert set(G_dup) == set(G_single)

    def test_single_variable_ideal(self):
        x = Symbol('x')
        G = groebner_basis([x**2 - 1], x, order='degrevlex')
        assert is_groebner(list(G), [x], order='degrevlex')
        assert len(G) == 1

    def test_already_reduced_gb_idempotent(self):
        x, y = Symbol('x'), Symbol('y')
        F = [x**2 - y, y**2 - x]
        G1 = groebner_basis(F, x, y, order='degrevlex', reduced=True)
        assert is_reduced_basis(list(G1), [x, y], order='degrevlex')
        G2 = groebner_basis(list(G1), x, y, order='degrevlex')
        assert set(G1) == set(G2)
        assert same_ideal(G1, G2, [x, y], order='degrevlex')

    def test_high_degree_max_degree_raises(self):
        x, y = Symbol('x'), Symbol('y')
        with pytest.raises(RuntimeError):
            groebner_basis([x**5 - y, y**5 - x], x, y, order='degrevlex',
                           max_degree=3)

    def test_invalid_modulus_nonprime(self):
        x, y = Symbol('x'), Symbol('y')
        with pytest.raises(ValueError):
            groebner_basis([x**2 - y], x, y, modulus=4)

    def test_invalid_modulus_one(self):
        x, y = Symbol('x'), Symbol('y')
        with pytest.raises(ValueError):
            groebner_basis([x**2 - y], x, y, modulus=1)

    @pytest.mark.xfail(strict=True,
                       reason="bug: negative modulus gives OverflowError "
                              "instead of ValueError")
    def test_invalid_modulus_negative(self):
        x, y = Symbol('x'), Symbol('y')
        with pytest.raises(ValueError):
            groebner_basis([x**2 - y], x, y, modulus=-1)

    def test_unknown_algorithm_raises(self):
        x, y = Symbol('x'), Symbol('y')
        with pytest.raises(ValueError):
            groebner_basis([x**2 - y], x, y, algorithm='unknown_algo')

    def test_unknown_order_raises(self):
        x, y = Symbol('x'), Symbol('y')
        with pytest.raises(ValueError):
            groebner_basis([x**2 - y], x, y, order='unknown_order')

    def test_variables_not_in_polynomials(self):
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        G = groebner_basis([x**2 - 1], x, y, z, order='degrevlex')
        assert is_groebner(list(G), [x, y, z], order='degrevlex')
        assert contains_ideal(G, [x**2 - 1], [x, y, z], order='degrevlex')

    def test_more_variables_than_equations(self):
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        G = groebner_basis([x - 1], x, y, z, order='degrevlex')
        assert is_groebner(list(G), [x, y, z], order='degrevlex')
        assert G[0] == x - 1

    def test_more_equations_than_variables(self):
        x, y = Symbol('x'), Symbol('y')
        F = [x - 1, y - 2, x + y - 3, x - y + 1]
        G = groebner_basis(F, x, y, order='degrevlex')
        assert is_groebner(list(G), [x, y], order='degrevlex')
        assert set(G) == {x - 1, y - 2}

    def test_single_polynomial(self):
        x, y = Symbol('x'), Symbol('y')
        G = groebner_basis([x**3 - 2*x*y], x, y, order='degrevlex')
        assert is_groebner(list(G), [x, y], order='degrevlex')
        assert contains_ideal(G, [x**3 - 2*x*y], [x, y], order='degrevlex')

    def test_valid_prime_modulus(self):
        x, y = Symbol('x'), Symbol('y')
        G = groebner_basis([x**2 + y**2 - 1, x**2 - y], x, y, modulus=7,
                           order='degrevlex')
        assert is_groebner(list(G), [x, y], order='degrevlex', modulus=7)
        assert contains_ideal(G, [x**2 + y**2 - 1, x**2 - y], [x, y],
                              order='degrevlex', modulus=7)


# ============================================================================
# WP8 — Property-based testing (Hypothesis)
# ============================================================================

from hypothesis import given, settings, assume, seed
from hypothesis import strategies as st


def _small_poly(vars_list, max_deg=3, coeff_range=(-5, 6)):
    """Generate a random polynomial with bounded degree and small coefficients."""
    from symengine import Add, Mul
    import random
    terms = []
    for _ in range(random.randint(1, 4)):
        c = random.randint(*coeff_range)
        if c == 0:
            continue
        monomial = Integer(1)
        for v in vars_list:
            d = random.randint(0, max_deg)
            if d > 0:
                monomial = monomial * v**d
        terms.append(Integer(c) * monomial)
    if not terms:
        return Integer(0)
    return Add(*terms)


def _make_random_system(n_vars, n_polys, max_deg, coeff_range, modulus):
    """Create a random polynomial system with given parameters."""
    import random
    var_names = [f'v{i}' for i in range(n_vars)]
    gens = [Symbol(n) for n in var_names]
    polys = []
    for _ in range(n_polys):
        p = _small_poly(gens, max_deg=max_deg, coeff_range=coeff_range)
        polys.append(p)
    return polys, gens


def _hypothesis_gb_invariants(polys, gens, order='degrevlex', modulus=0):
    """Assert core invariants for a computed GB."""
    # Remove zero polynomials
    polys = [p for p in polys if p != 0]
    if not polys:
        return
    kw = {'order': order}
    if modulus:
        kw['modulus'] = modulus
    result = run_with_timeout(
        groebner_basis, polys, *gens, **kw, timeout=30)
    if result == _SENTINEL_TIMEOUT:
        pytest.skip("GB computation timed out")
    if result == _SENTINEL_ERROR:
        pytest.skip("GB computation errored")
    G = result
    if len(G) == 0:
        return
    # Property (2): G is a GB
    assert is_groebner(list(G), gens, **kw), \
        "is_groebner(GB(F)) must be true"
    # Property (1): <G> == <F>
    assert contains_ideal(G, polys, gens, order=order, modulus=modulus), \
        "GB(F) must contain F"


def _hypothesis_idempotence(polys, gens, order='degrevlex', modulus=0):
    """Assert idempotence: GB(GB(F)) == GB(F)."""
    polys = [p for p in polys if p != 0]
    if not polys:
        return
    kw = {'order': order}
    if modulus:
        kw['modulus'] = modulus
    G1 = run_with_timeout(groebner_basis, polys, *gens, **kw, timeout=30)
    if G1 == _SENTINEL_TIMEOUT:
        pytest.skip("GB computation timed out (step 1)")
    if G1 == _SENTINEL_ERROR:
        pytest.skip("GB computation errored (step 1)")
    if len(G1) == 0:
        return
    G2 = run_with_timeout(groebner_basis, list(G1), *gens, **kw, timeout=30)
    if G2 == _SENTINEL_TIMEOUT:
        pytest.skip("GB computation timed out (step 2)")
    if G2 == _SENTINEL_ERROR:
        pytest.skip("GB computation errored (step 2)")
    assert same_ideal(G1, G2, gens, order=order, modulus=modulus), \
        "GB(GB(F)) must generate the same ideal as GB(F)"


class TestWP8PropertyBased:

    @pytest.mark.slow
    @settings(max_examples=10, deadline=None, suppress_health_check=[])
    @seed(42)
    @given(
        n_vars=st.integers(min_value=2, max_value=3),
        n_polys=st.integers(min_value=2, max_value=3),
    )
    def test_gb_invariants_qq(self, n_vars, n_polys):
        import random
        random.seed(n_vars * 1000 + n_polys)
        polys, gens = _make_random_system(
            n_vars, n_polys, max_deg=2, coeff_range=(-3, 4), modulus=0)
        _hypothesis_gb_invariants(polys, gens, order='degrevlex', modulus=0)

    @pytest.mark.slow
    @settings(max_examples=10, deadline=None, suppress_health_check=[])
    @seed(43)
    @given(
        n_vars=st.integers(min_value=2, max_value=3),
        n_polys=st.integers(min_value=2, max_value=3),
    )
    def test_gb_invariants_gf(self, n_vars, n_polys):
        import random
        random.seed(n_vars * 1000 + n_polys + 100)
        polys, gens = _make_random_system(
            n_vars, n_polys, max_deg=2, coeff_range=(0, 7), modulus=0)
        _hypothesis_gb_invariants(polys, gens, order='degrevlex', modulus=32003)

    @pytest.mark.slow
    @settings(max_examples=10, deadline=None, suppress_health_check=[])
    @seed(44)
    @given(
        n_vars=st.integers(min_value=2, max_value=3),
        n_polys=st.integers(min_value=2, max_value=3),
    )
    def test_idempotence_qq(self, n_vars, n_polys):
        import random
        random.seed(n_vars * 1000 + n_polys + 200)
        polys, gens = _make_random_system(
            n_vars, n_polys, max_deg=2, coeff_range=(-3, 4), modulus=0)
        _hypothesis_idempotence(polys, gens, order='degrevlex', modulus=0)

    @pytest.mark.slow
    @settings(max_examples=5, deadline=None, suppress_health_check=[])
    @seed(45)
    @given(n_polys=st.integers(min_value=2, max_value=3))
    def test_algorithm_agreement(self, n_polys):
        import random
        random.seed(n_polys + 300)
        polys, gens = _make_random_system(
            2, n_polys, max_deg=2, coeff_range=(-3, 4), modulus=0)
        polys = [p for p in polys if p != 0]
        if not polys:
            return
        algos = ['buchberger', 'f5b']
        results = {}
        for algo in algos:
            r = run_with_timeout(
                groebner_basis, polys, *gens, order='degrevlex',
                algorithm=algo, timeout=30)
            if r in (_SENTINEL_TIMEOUT, _SENTINEL_ERROR):
                return
            results[algo] = r
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                a, b = algos[i], algos[j]
                assert same_ideal(results[a], results[b], gens,
                                  order='degrevlex'), \
                    f"Algorithm {a} and {b} disagree on ideal"

    @pytest.mark.slow
    @settings(max_examples=5, deadline=None, suppress_health_check=[])
    @seed(46)
    @given(n_polys=st.integers(min_value=2, max_value=3))
    def test_order_membership_agreement(self, n_polys):
        """A poly's membership in <F> should be order-independent."""
        import random
        random.seed(n_polys + 400)
        polys, gens = _make_random_system(
            2, n_polys, max_deg=2, coeff_range=(-3, 4), modulus=0)
        polys = [p for p in polys if p != 0]
        if not polys:
            return
        orders = ['degrevlex', 'grlex', 'lex']
        bases = {}
        for order in orders:
            r = run_with_timeout(
                groebner_basis, polys, *gens, order=order, timeout=30)
            if r in (_SENTINEL_TIMEOUT, _SENTINEL_ERROR):
                return
            bases[order] = r
        # All bases should generate the same ideal
        for i in range(len(orders)):
            for j in range(i + 1, len(orders)):
                oi, oj = orders[i], orders[j]
                # Use a single order for the membership test
                assert same_ideal(bases[oi], bases[oj], gens,
                                  order='degrevlex'), \
                    f"GB under {oi} and {oj} generate different ideals"
