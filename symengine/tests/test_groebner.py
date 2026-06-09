from symengine import Symbol, groebner_basis, GroebnerBasis, Integer, Rational, groebner
from symengine.lib.symengine_wrapper import (
    normal_form, is_groebner, is_reduced_basis, solve_poly_system
)


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
    assert len(G) > 0


def test_groebner_cyclic4():
    a, b, c, d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')
    f1 = a + b + c + d
    f2 = a*b + b*c + c*d + d*a
    f3 = a*b*c + b*c*d + c*d*a + d*a*b
    f4 = a*b*c*d - 1
    G = groebner_basis([f1, f2, f3, f4], a, b, c, d, order='degrevlex')
    assert len(G) > 0


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


def test_solve_poly_system_basic():
    x, y = Symbol('x'), Symbol('y')
    solutions = solve_poly_system([x**2 + y**2 - 1, x - y], x, y)
    assert solutions is not None


def test_solve_poly_system_trivial():
    x, y = Symbol('x'), Symbol('y')
    solutions = solve_poly_system([x - 1, y - 2], x, y)
    assert solutions is not None


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
    assert len(G) > 0


def test_groebner_stats():
    x, y = Symbol('x'), Symbol('y')
    f1 = x**2 + y**2 - 1
    f2 = x - y
    G = groebner_basis([f1, f2], x, y)
    stats = G.stats
    assert 'input_polys' in stats
    assert 'output_polys' in stats
    assert stats['input_polys'] == 2
