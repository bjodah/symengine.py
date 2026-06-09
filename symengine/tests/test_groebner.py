import pytest

from symengine import (
    Symbol, groebner_basis, GroebnerBasis, Integer, Rational, groebner,
    FiniteSet, EmptySet, sympify,
)
from symengine.lib.symengine_wrapper import (
    normal_form, is_groebner, is_reduced_basis, is_zero_dimensional,
    solve_poly_system
)
from symengine.tests.groebner_corpus import SYSTEMS


def _same_ideal(A, B, gens, order):
    """Two bases generate the same ideal iff every element of each reduces to
    zero modulo the other. Robust to representation differences."""
    A, B = list(A), list(B)
    return (all(normal_form(p, B, gens, order=order) == 0 for p in A) and
            all(normal_form(p, A, gens, order=order) == 0 for p in B))


def _assert_groebner(G, gens, order, modulus=0):
    """Every basis we compute must actually be a Groebner basis. This is the
    single assertion that would have caught the MoGVW defect."""
    assert len(G) > 0
    assert is_groebner(list(G), gens, order=order, modulus=modulus) is True


def test_groebner_basic_lex():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + 2*x*y**2
    f2 = x*y + 2*y**3 - 1
    G = groebner_basis([f1, f2], x, y, order='lex')
    assert len(G) == 2
    assert G[0] == x
    assert G[1] == y**3 - Rational(1, 2)


def test_groebner_basic_grlex():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**3 - 2*x*y
    f2 = x**2*y + x - 2*y**2
    G = groebner_basis([f1, f2], x, y, order='grlex')
    assert len(G) > 0
    assert is_groebner(list(G), [x, y], order='grlex')


def test_groebner_basic_degrevlex():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + 2*x*y**2
    f2 = x*y + 2*y**3 - 1
    G = groebner_basis([f1, f2], x, y, order='degrevlex')
    assert len(G) > 0
    assert is_groebner(list(G), [x, y], order='degrevlex')


def test_groebner_three_vars():
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    f1 = x - z**2
    f2 = y - z**3
    G = groebner_basis([f1, f2], x, y, z, order='lex')
    assert len(G) == 2


def test_groebner_elimination():
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    f1 = -x**2 + y
    f2 = -x**3 + z
    G = groebner_basis([f1, f2], x, y, z, order='lex')
    assert len(G) == 4


def test_groebner_properties():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y**2 - 1
    f2 = x - y
    G = groebner_basis([f1, f2], x, y, order='degrevlex')
    assert isinstance(G, GroebnerBasis)
    assert len(G) > 0
    assert G.order == 'degrevlex'
    assert G.modulus == 0
    assert isinstance(G.stats, dict)
    assert isinstance(G.polys, tuple)
    assert isinstance(G.gens, tuple)
    assert list(G.gens) == [x, y]


def test_groebner_repr():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y
    G = groebner_basis([f1], x, y, order='lex')
    r = repr(G)
    assert 'GroebnerBasis' in r
    assert 'lex' in r


def test_groebner_eq():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y**2 - 1
    f2 = x - y
    G1 = groebner_basis([f1, f2], x, y, order='degrevlex')
    G2 = groebner_basis([f1, f2], x, y, order='degrevlex')
    assert G1 == G2


def test_groebner_iter():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y
    G = groebner_basis([f1], x, y)
    polys = list(G)
    assert len(polys) == len(G)
    for i, p in enumerate(G):
        assert p == G[i]


def test_groebner_single_polynomial():
    x = Symbol('x')
    f = x**2 - 1
    G = groebner_basis([f], x, order='degrevlex')
    assert len(G) == 1


def test_groebner_trivial_ideal():
    x, y = Symbol('x'), Symbol('y')
    G = groebner_basis([x, x - 1], x, y, order='degrevlex')
    assert len(G) == 1
    assert G[0] == 1


def test_groebner_katsura3():
    x0, x1, x2 = Symbol('x0'), Symbol('x1'), Symbol('x2')
    f1 = x0 + 2*x1 + 2*x2 - 1
    f2 = x0**2 + 2*x1**2 + 2*x2**2 - x0
    f3 = 2*x0*x1 + 2*x1*x2 - x1
    G_lex = groebner_basis([f1, f2, f3], x0, x1, x2, order='lex')
    assert len(G_lex) > 0
    assert is_groebner(list(G_lex), [x0, x1, x2], order='lex')
    G_grlex = groebner_basis([f1, f2, f3], x0, x1, x2, order='grlex')
    assert len(G_grlex) > 0
    assert is_groebner(list(G_grlex), [x0, x1, x2], order='grlex')


def test_groebner_katsura4():
    x0, x1, x2, x3 = Symbol('x0'), Symbol('x1'), Symbol('x2'), Symbol('x3')
    f1 = x0 + 2*x1 + 2*x2 + 2*x3 - 1
    f2 = x0**2 + 2*x1**2 + 2*x2**2 + 2*x3**2 - x0
    f3 = 2*x0*x1 + 2*x1*x2 + 2*x2*x3 - x1
    f4 = x1**2 + 2*x0*x2 + 2*x1*x3 - x2
    G = groebner_basis([f1, f2, f3, f4], x0, x1, x2, x3, order='degrevlex')
    _assert_groebner(G, [x0, x1, x2, x3], 'degrevlex')
    assert len(G) == 7


def test_groebner_cyclic4():
    a, b, c, d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')
    f1 = a + b + c + d
    f2 = a*b + b*c + c*d + d*a
    f3 = a*b*c + b*c*d + c*d*a + d*a*b
    f4 = a*b*c*d - 1
    G = groebner_basis([f1, f2, f3, f4], a, b, c, d, order='degrevlex')
    _assert_groebner(G, [a, b, c, d], 'degrevlex')
    assert len(G) == 7


def test_groebner_algorithm_consistency():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + 2*x*y**2
    f2 = x*y + 2*y**3 - 1
    G_buch = groebner_basis([f1, f2], x, y, order='degrevlex', algorithm='buchberger')
    G_f5b = groebner_basis([f1, f2], x, y, order='degrevlex', algorithm='f5b')
    assert is_groebner(list(G_buch), [x, y], order='degrevlex')
    assert is_groebner(list(G_f5b), [x, y], order='degrevlex')


def test_is_groebner_positive():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + 2*x*y**2
    f2 = x*y + 2*y**3 - 1
    G = groebner_basis([f1, f2], x, y, order='degrevlex')
    assert is_groebner(list(G), [x, y], order='degrevlex') is True


def test_is_groebner_negative():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + 2*x*y**2
    f2 = x*y + 2*y**3 - 1
    assert is_groebner([f1, f2], [x, y], order='degrevlex') is False


def test_is_reduced_basis():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + 2*x*y**2
    f2 = x*y + 2*y**3 - 1
    G = groebner_basis([f1, f2], x, y, order='degrevlex', reduced=True)
    assert is_reduced_basis(list(G), [x, y], order='degrevlex') is True


def test_normal_form_basic():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y
    f2 = x*y + 1
    G = groebner_basis([f1, f2], x, y, order='degrevlex')
    rem = normal_form(x**3 + x*y, list(G), [x, y], order='degrevlex')
    assert rem is not None


def test_normal_form_groebner_basis_input():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 - 1
    f2 = y**2 - 1
    G = groebner_basis([f1, f2], x, y, order='degrevlex')
    rem = normal_form(x*y + 1, G, [x, y])
    assert rem is not None


def test_groebner_basis_reduce_method():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 - 1
    f2 = y**2 - 1
    G = groebner_basis([f1, f2], x, y, order='degrevlex')
    rem = G.reduce(x*y + 1)
    assert rem is not None


def test_solve_poly_system_linear():
    # {2x-3, 3y/2 - 2x, z - 5y} has the unique solution (3/2, 2, 10).
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    sol = solve_poly_system([2*x - 3, Rational(3, 2)*y - 2*x, z - 5*y], x, y, z)
    assert isinstance(sol, FiniteSet)
    points = list(sol.args)
    assert len(points) == 1
    # Each solution point is an accessible Tuple of components (regression test
    # for the Tuple/c2py wiring: this used to raise "Unsupported SymEngine class").
    assert list(points[0].args) == [Rational(3, 2), Integer(2), Integer(10)]


def test_solve_poly_system_trivial():
    x, y = Symbol('x'), Symbol('y')
    sol = solve_poly_system([x - 1, y - 2], x, y)
    assert isinstance(sol, FiniteSet)
    points = list(sol.args)
    assert len(points) == 1
    assert list(points[0].args) == [Integer(1), Integer(2)]


def test_solve_poly_system_finite_solutions():
    # {x^2 - y, x^3 - x} has solutions (0,0), (1,1), (-1,1) -> 3 points.
    x, y = Symbol('x'), Symbol('y')
    sol = solve_poly_system([x**2 - y, x**3 - x], x, y)
    assert isinstance(sol, FiniteSet)
    points = {tuple(p.args) for p in sol.args}
    assert points == {
        (Integer(0), Integer(0)),
        (Integer(1), Integer(1)),
        (Integer(-1), Integer(1)),
    }


def test_solve_poly_system_inconsistent():
    # {y - x, y - x - 1} is inconsistent -> empty solution set.
    x, y = Symbol('x'), Symbol('y')
    sol = solve_poly_system([y - x, y - x - 1], x, y)
    assert sol == EmptySet


def test_fglm_conversion():
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    f1 = x**2 + y + z - 1
    f2 = x + y**2 + z - 1
    f3 = x + y + z**2 - 1
    G_drl = groebner_basis([f1, f2, f3], x, y, z, order='degrevlex')
    G_lex = G_drl.fglm('lex')
    assert G_lex.order == 'lex'
    assert is_groebner(list(G_lex), [x, y, z], order='lex')


def test_groebner_alias():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y
    G = groebner([f1], x, y, order='degrevlex')
    assert isinstance(G, GroebnerBasis)
    assert len(G) > 0


def test_groebner_default_order():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y**2 - 1
    f2 = x - y
    G = groebner_basis([f1, f2], x, y)
    assert G.order == 'degrevlex'


def test_groebner_input_validation():
    x = Symbol('x')
    try:
        groebner_basis([], x)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    try:
        groebner_basis([x])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_groebner_gf():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y**2 - 1
    f2 = x**2 - y
    G = groebner_basis([f1, f2], x, y, modulus=7, order='degrevlex')
    _assert_groebner(G, [x, y], 'degrevlex', modulus=7)


def test_groebner_stats():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y**2 - 1
    f2 = x - y
    G = groebner_basis([f1, f2], x, y)
    stats = G.stats
    assert 'input_polys' in stats
    assert 'output_polys' in stats
    assert stats['input_polys'] == 2


def test_groebner_exact_lex_basis():
    # Verified canonical answer from the C++ test suite: {x, y^3 - 1/2}.
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + 2*x*y**2
    f2 = x*y + 2*y**3 - 1
    G = groebner_basis([f1, f2], x, y, order='lex')
    assert is_groebner(list(G), [x, y], order='lex')
    assert is_reduced_basis(list(G), [x, y], order='lex')
    assert set(G) == {x, y**3 - Rational(1, 2)}


def test_groebner_grevlex_alias():
    # 'grevlex' (SymPy spelling) must be accepted as 'degrevlex'.
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + 2*x*y**2
    f2 = x*y + 2*y**3 - 1
    G1 = groebner_basis([f1, f2], x, y, order='grevlex')
    G2 = groebner_basis([f1, f2], x, y, order='degrevlex')
    assert G1.order == 'degrevlex'
    assert set(G1) == set(G2)


def test_groebner_cross_order_same_ideal():
    # lex and grlex bases of the same input describe the same ideal.
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    f1 = -x**2 + y
    f2 = -x**3 + z
    G_lex = groebner_basis([f1, f2], x, y, z, order='lex')
    G_grlex = groebner_basis([f1, f2], x, y, z, order='grlex')
    assert _same_ideal(G_lex, G_grlex, [x, y, z], order='lex')


def test_groebner_cross_algorithm_same_ideal():
    x, y = Symbol('x'), Symbol('y')
    f1, f2 = x**2 - y, x**3 - x
    G_buch = groebner_basis([f1, f2], x, y, order='degrevlex', algorithm='buchberger')
    G_f5b = groebner_basis([f1, f2], x, y, order='degrevlex', algorithm='f5b')
    G_mogvw = groebner_basis([f1, f2], x, y, order='degrevlex', algorithm='mogvw')
    assert _same_ideal(G_buch, G_f5b, [x, y], order='degrevlex')
    assert _same_ideal(G_buch, G_mogvw, [x, y], order='degrevlex')


def _katsura4():
    x0, x1, x2, x3 = Symbol('x0'), Symbol('x1'), Symbol('x2'), Symbol('x3')
    f1 = x0 + 2*x1 + 2*x2 + 2*x3 - 1
    f2 = x0**2 + 2*x1**2 + 2*x2**2 + 2*x3**2 - x0
    f3 = 2*x0*x1 + 2*x1*x2 + 2*x2*x3 - x1
    f4 = x1**2 + 2*x0*x2 + 2*x1*x3 - x2
    return [f1, f2, f3, f4], [x0, x1, x2, x3]


@pytest.mark.parametrize("algorithm", ["buchberger", "f5b", "mogvw"])
def test_groebner_katsura4_correct(algorithm):
    polys, gens = _katsura4()
    G = groebner_basis(polys, *gens, order='degrevlex', algorithm=algorithm)
    _assert_groebner(G, gens, 'degrevlex')
    assert len(G) == 7


def test_groebner_katsura4_mogvw_correct():
    polys, gens = _katsura4()
    G = groebner_basis(polys, *gens, order='degrevlex', algorithm='mogvw')
    _assert_groebner(G, gens, 'degrevlex')
    assert len(G) == 7


def test_groebner_katsura4_gf_grlex_auto_correct():
    polys, gens = _katsura4()
    G = groebner_basis(polys, *gens, order='grlex', modulus=32003)
    assert G.algorithm != 'mogvw'      # AUTO does not select MoGVW
    _assert_groebner(G, gens, 'grlex', modulus=32003)


def test_groebner_fractional_field_parametric():
    x, y = Symbol('x'), Symbol('y')
    C1, C2, C3 = Symbol('C1'), Symbol('C2'), Symbol('C3')
    polys = [C1*x*y + C2*y**2 - 1, x**2 - C3*y]
    gens = [x, y]
    for algorithm in ('auto', 'buchberger', 'f5b'):
        G = groebner_basis(polys, *gens, order='degrevlex', algorithm=algorithm)
        assert len(G) > 0
        assert is_groebner(list(G), gens, order='degrevlex') is True
    G_buch = groebner_basis(polys, *gens, order='degrevlex', algorithm='buchberger')
    G_f5b = groebner_basis(polys, *gens, order='degrevlex', algorithm='f5b')
    assert _same_ideal(G_buch, G_f5b, gens, order='degrevlex')


def test_groebner_cross_algorithm_katsura4_mogvw_same_ideal():
    polys, gens = _katsura4()
    G_buch = groebner_basis(polys, *gens, order='degrevlex', algorithm='buchberger')
    G_mogvw = groebner_basis(polys, *gens, order='degrevlex', algorithm='mogvw')
    _assert_groebner(G_mogvw, gens, 'degrevlex')
    assert len(G_mogvw) == 7
    assert _same_ideal(G_buch, G_mogvw, gens, order='degrevlex')


def test_normal_form_ideal_membership():
    x, y = Symbol('x'), Symbol('y')
    G = groebner_basis([x**2 - 1, y**2 - 1], x, y, order='degrevlex')
    # x^2 - 1 is in the ideal -> reduces to 0; x*y + 1 is not.
    assert normal_form(x**2 - 1, list(G), [x, y], order='degrevlex') == 0
    assert normal_form(x*y + 1, list(G), [x, y], order='degrevlex') != 0


def test_fglm_non_zero_dimensional_raises():
    # {x*y - 1} over (x, y) is not zero-dimensional -> FGLM must reject it.
    x, y = Symbol('x'), Symbol('y')
    G = groebner_basis([x*y - 1], x, y, order='degrevlex')
    try:
        G.fglm('lex')
        assert False, "expected FGLM to reject a non-zero-dimensional ideal"
    except NotImplementedError:
        pass


def test_groebner_invalid_modulus():
    # Composite / invalid moduli are rejected (UnsupportedCoefficientDomain).
    x, y = Symbol('x'), Symbol('y')
    for bad in (1, 4):
        try:
            groebner_basis([x**2 - y], x, y, modulus=bad)
            assert False, "expected ValueError for modulus=%d" % bad
        except ValueError:
            pass
    # Valid prime modulus succeeds.
    G = groebner_basis([x**2 - y], x, y, modulus=7)
    assert len(G) > 0


def test_groebner_max_degree_limit():
    # Input degree 5 with max_degree=3 must trip the resource limit.
    x, y = Symbol('x'), Symbol('y')
    try:
        groebner_basis([x**5 - y, y**5 - x], x, y, order='degrevlex', max_degree=3)
        assert False, "expected RuntimeError for exceeded max_degree"
    except RuntimeError:
        pass


def test_groebner_alias_sympy_defaults():
    # The SymPy-compatible groebner() defaults to lex (unlike groebner_basis).
    x, y = Symbol('x'), Symbol('y')
    G = groebner([x**2 + 2*x*y**2, x*y + 2*y**3 - 1], x, y)
    assert G.order == 'lex'
    # SymPy's method= keyword maps to algorithm=.
    G2 = groebner([x**2 - y, x**3 - x], x, y, method='f5b')
    assert G2.algorithm == 'f5b'


def test_is_zero_dimensional_property():
    x, y = Symbol('x'), Symbol('y')
    # {x^2 - y, y^2 - x}: finitely many solutions -> zero-dimensional.
    G0 = groebner_basis([x**2 - y, y**2 - x], x, y, order='degrevlex')
    assert G0.is_zero_dimensional is True
    # {x*y - 1}: a curve -> not zero-dimensional.
    G1 = groebner_basis([x*y - 1], x, y, order='degrevlex')
    assert G1.is_zero_dimensional is False


def test_is_zero_dimensional_function():
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    G = groebner_basis([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1],
                       x, y, z, order='degrevlex')
    assert is_zero_dimensional(list(G), [x, y, z], order='degrevlex') is True
    # An unconstrained extra variable makes it positive-dimensional.
    t = Symbol('t')
    G2 = groebner_basis([x, y, z], x, y, z, t, order='degrevlex')
    assert is_zero_dimensional(list(G2), [x, y, z, t], order='degrevlex') is False


def test_is_zero_dimensional_matches_fglm():
    # The predicate must agree with whether fglm() succeeds.
    x, y = Symbol('x'), Symbol('y')
    G = groebner_basis([x*y - 1], x, y, order='degrevlex')
    assert G.is_zero_dimensional is False
    try:
        G.fglm('lex')
        assert False, "fglm should reject a non-zero-dimensional ideal"
    except NotImplementedError:
        pass


def test_fglm_round_trip():
    # degrevlex -> lex -> degrevlex preserves the ideal.
    x, y = Symbol('x'), Symbol('y')
    G = groebner_basis([x**2 - y, y**2 - x], x, y, order='degrevlex')
    G_lex = G.fglm('lex')
    assert G_lex.order == 'lex'
    assert is_groebner(list(G_lex), [x, y], order='lex')
    G_back = G_lex.fglm('degrevlex')
    assert _same_ideal(G, G_back, [x, y], order='degrevlex')


def test_groebner_extended_stats():
    x, y = Symbol('x'), Symbol('y')
    G = groebner_basis([x**2 + 2*x*y**2, x*y + 2*y**3 - 1], x, y, order='degrevlex')
    stats = G.stats
    # Signature-based and matrix counters are surfaced.
    for key in ('f5_reductions', 'rejected_by_syzygy', 'matrices_built',
                'rows_reduced_to_zero', 'genericity_assumptions'):
        assert key in stats
    # genericity_assumptions is a tuple of expressions (empty unless the
    # computation made parametric assumptions).
    assert isinstance(stats['genericity_assumptions'], tuple)


def test_compare_with_sympy():
    sympy = pytest.importorskip("sympy")
    sx, sy, sz = sympy.symbols('x y z')
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

    cases = [
        ([sx**2 + 2*sx*sy**2, sx*sy + 2*sy**3 - 1], [sx, sy], [x, y], 'lex'),
        ([sx**3 - 2*sx*sy, sx**2*sy + sx - 2*sy**2], [sx, sy], [x, y], 'grlex'),
        ([-sx**2 + sy, -sx**3 + sz], [sx, sy, sz], [x, y, z], 'lex'),
    ]
    for sympy_polys, sympy_gens, se_gens, order in cases:
        sp_G = sympy.groebner(sympy_polys, *sympy_gens, order=order)
        se_polys = [sympify(p) for p in sympy_polys]
        se_G = groebner_basis(se_polys, *se_gens, order=order)
        # Same number of reduced generators.
        assert len(se_G) == len(sp_G)
        # Every SymEngine basis element lies in the SymPy ideal and vice versa.
        for p in se_G:
            _, r = sympy.reduced(p._sympy_(), list(sp_G.exprs), *sympy_gens, order=order)
            assert r == 0
        for q in sp_G.exprs:
            assert normal_form(sympify(q), list(se_G), se_gens, order=order) == 0


def test_groebner_eq_distinguishes_modulus():
    x, y = Symbol('x'), Symbol('y')
    f1, f2 = x**2 - y, y**2 - x
    G_qq = groebner_basis([f1, f2], x, y, order='degrevlex')
    G_gf = groebner_basis([f1, f2], x, y, order='degrevlex', modulus=7)
    # Same polys/gens/order but different ground field -> not equal.
    assert G_qq != G_gf
    assert G_qq == groebner_basis([f1, f2], x, y, order='degrevlex')


def test_groebner_reduce_matches_normal_form():
    x, y = Symbol('x'), Symbol('y')
    G = groebner_basis([x**2 - 1, y**2 - 1], x, y, order='degrevlex')
    assert G.reduce(x*y + 1) == normal_form(x*y + 1, list(G), [x, y],
                                            order='degrevlex')


def test_groebner_parametric_genericity_assumptions_contract():
    x, y = Symbol('x'), Symbol('y')
    C1, C2 = Symbol('C1'), Symbol('C2')
    G = groebner_basis([C1*x**2 + C2*y, x*y - 1], x, y, order='degrevlex')
    assert len(G) > 0
    # Currently always empty; this test documents that contract and must be
    # updated if/when real tracking is implemented.
    assert isinstance(G.stats['genericity_assumptions'], tuple)


# ---------------------------------------------------------------------------
# Vendored corpus tests (from groebner_corpus.py)
# ---------------------------------------------------------------------------

def _build(name):
    """Build gens/polys/golden_size from a corpus entry."""
    s = SYSTEMS[name]
    gens = [Symbol(g) for g in s["gens"]]
    env = {g: gens[i] for i, g in enumerate(s["gens"])}
    polys = [eval(p, {"__builtins__": {}}, env) for p in s["polys"]]
    return polys, gens, s["golden_size"]


_FAST = ["cyclic3", "cyclic4"]
_SLOW = [n for n in SYSTEMS if n not in _FAST]


@pytest.mark.parametrize("name", _FAST)
@pytest.mark.parametrize("algorithm", ["buchberger", "f5b", "mogvw"])
def test_corpus_correct_degrevlex(name, algorithm):
    polys, gens, golden = _build(name)
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm=algorithm)
    assert is_groebner(list(G), gens, order="degrevlex") is True
    assert len(G) == golden


@pytest.mark.slow
@pytest.mark.parametrize("name", _SLOW)
def test_corpus_correct_degrevlex_slow(name):
    polys, gens, golden = _build(name)
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b")
    assert is_groebner(list(G), gens, order="degrevlex") is True
    assert len(G) == golden


@pytest.mark.parametrize("name", _FAST)
def test_corpus_cross_algorithm_same_ideal(name):
    polys, gens, _ = _build(name)
    Gb = groebner_basis(polys, *gens, order="degrevlex", algorithm="buchberger")
    Gm = groebner_basis(polys, *gens, order="degrevlex", algorithm="mogvw")
    assert _same_ideal(Gb, Gm, gens, order="degrevlex")


@pytest.mark.parametrize("name", _FAST)
def test_corpus_gf_prime(name):
    polys, gens, _ = _build(name)
    G = groebner_basis(polys, *gens, order="degrevlex", modulus=32003)
    assert is_groebner(list(G), gens, order="degrevlex", modulus=32003) is True


# ---------------------------------------------------------------------------
# Exact golden bases (ported from moGVWTest.cpp)
# ---------------------------------------------------------------------------

def test_corpus_cyclic3_exact_degrevlex():
    a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
    polys = [a*b*c - 1, a*b + b*c + c*a, a + b + c]
    G = groebner_basis(polys, a, b, c, order='degrevlex')
    _assert_groebner(G, [a, b, c], 'degrevlex')
    assert set(G) == {a + b + c, b**2 + b*c + c**2, c**3 - 1}


def test_corpus_cyclic3_exact_lex():
    a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
    polys = [a*b*c - 1, a*b + b*c + c*a, a + b + c]
    G = groebner_basis(polys, a, b, c, order='lex')
    _assert_groebner(G, [a, b, c], 'lex')
    assert set(G) == {a + b + c, b**2 + b*c + c**2, c**3 - 1}


def test_corpus_cyclic4_exact_degrevlex():
    a, b, c, d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')
    polys = [a*b*c*d - 1, a*b*c + a*b*d + a*c*d + b*c*d,
             a*b + b*c + a*d + c*d, a + b + c + d]
    G = groebner_basis(polys, a, b, c, d, order='degrevlex')
    _assert_groebner(G, [a, b, c, d], 'degrevlex')
    assert len(G) == 7
    assert set(G) == {
        a + b + c + d,
        b**2 + 2*b*d + d**2,
        b*c**2 + c**2*d - b*d**2 - d**3,
        b*c*d**2 + c**2*d**2 - b*d**3 + c*d**3 - d**4 - 1,
        b*d**4 + d**5 - b - d,
        c**3*d**2 + c**2*d**3 - c - d,
        c**2*d**4 + b*c - b*d + c*d - 2*d**2,
    }


def test_corpus_cyclic4_exact_lex():
    a, b, c, d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')
    polys = [a*b*c*d - 1, a*b*c + a*b*d + a*c*d + b*c*d,
             a*b + b*c + a*d + c*d, a + b + c + d]
    G = groebner_basis(polys, a, b, c, d, order='lex')
    _assert_groebner(G, [a, b, c, d], 'lex')
    assert len(G) == 6
    assert set(G) == {
        c**2*d**6 - c**2*d**2 - d**4 + 1,
        c**3*d**2 + c**2*d**3 - c - d,
        b*d**4 - b + d**5 - d,
        b*c - b*d + c**2*d**4 + c*d - 2*d**2,
        b**2 + 2*b*d + d**2,
        a + b + c + d,
    }


# ---------------------------------------------------------------------------
# Coverage gap tests: solve_poly_system, fglm, normal_form
# ---------------------------------------------------------------------------

def test_corpus_solve_poly_system():
    x, y = Symbol('x'), Symbol('y')
    sol = solve_poly_system([x**2 - y, x**3 - x], x, y)
    assert isinstance(sol, FiniteSet)
    points = {tuple(p.args) for p in sol.args}
    assert points == {
        (Integer(0), Integer(0)),
        (Integer(1), Integer(1)),
        (Integer(-1), Integer(1)),
    }


def test_corpus_fglm_round_trip():
    x0, x1, x2, x3 = Symbol('x0'), Symbol('x1'), Symbol('x2'), Symbol('x3')
    f1 = x0 + 2*x1 + 2*x2 + 2*x3 - 1
    f2 = x0**2 + 2*x1**2 + 2*x2**2 + 2*x3**2 - x0
    f3 = 2*x0*x1 + 2*x1*x2 + 2*x2*x3 - x1
    f4 = x1**2 + 2*x0*x2 + 2*x1*x3 - x2
    G = groebner_basis([f1, f2, f3, f4], x0, x1, x2, x3, order='degrevlex')
    G_lex = G.fglm('lex')
    assert G_lex.order == 'lex'
    assert is_groebner(list(G_lex), [x0, x1, x2, x3], order='lex')
    G_back = G_lex.fglm('degrevlex')
    assert is_groebner(list(G_back), [x0, x1, x2, x3], order='degrevlex')
    assert _same_ideal(G, G_back, [x0, x1, x2, x3], order='degrevlex')


def test_corpus_normal_form_membership():
    a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
    polys = [a*b*c - 1, a*b + b*c + c*a, a + b + c]
    G = groebner_basis(polys, a, b, c, order='degrevlex')
    G_list = list(G)
    gens = [a, b, c]
    # Each generator reduces to 0 modulo the basis.
    for p in polys:
        assert normal_form(p, G_list, gens, order='degrevlex') == 0
    # A perturbation not in the ideal does not reduce to 0.
    assert normal_form(a + 1, G_list, gens, order='degrevlex') != 0
