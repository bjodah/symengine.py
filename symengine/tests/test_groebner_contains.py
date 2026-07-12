"""P2-8a: GroebnerBasis.contains() -- SymPy parity for ideal membership.

SymPy's GroebnerBasis.contains(f) has no counterpart in the wrapper before
this; the underlying capability already exists as normal_form(f, G, gens)
== 0. contains() is implemented as self.reduce(f) == 0, reusing reduce()'s
existing options/basis plumbing.

Related, deliberately undone SymPy divergences (documented, not
implemented): our reduce() returns the remainder only (no quotient list),
and SymPy's strict= kwarg for unsolvable factors has no equivalent here (we
report outcome='failed'/EmptySet from solve_poly_system_ex instead).
"""

import pytest

from symengine import Symbol, groebner_basis, Integer

x, y = Symbol('x'), Symbol('y')


def test_contains_member_qq():
    # basis of {x**2 + y, x*y - 1}; multiply generators by monomials and add
    # to build a provable ideal member.
    G = groebner_basis([x**2 + y, x * y - 1], x, y, order='degrevlex')
    member = x * (x**2 + y) + y**2 * (x * y - 1)
    assert G.contains(member) is True


def test_contains_non_member_qq():
    G = groebner_basis([x**2 + y, x * y - 1], x, y, order='degrevlex')
    assert G.contains(Integer(1)) is False
    assert G.contains(x) is False


def test_contains_zero_always_true():
    G = groebner_basis([x**2 + y, x * y - 1], x, y, order='degrevlex')
    assert G.contains(0) is True
    assert G.contains(Integer(0)) is True


def test_contains_one_not_in_proper_ideal():
    G = groebner_basis([x**2 - 1, y**2 - 1], x, y, order='degrevlex')
    assert G.contains(1) is False


def test_contains_gfp_member():
    G = groebner_basis([x**2 + y, x * y - 1], x, y, order='degrevlex',
                       modulus=32003)
    member = x * (x**2 + y) + y**2 * (x * y - 1)
    assert G.contains(member) is True
    assert G.contains(1) is False


def test_contains_unit_ideal_everything_contained():
    G = groebner_basis([x, 1], x, y, order='degrevlex')
    assert list(G.polys) == [Integer(1)]
    assert G.contains(1) is True
    assert G.contains(x + y) is True
    assert G.contains(x**5 * y**3 - 7) is True
    assert G.contains(0) is True


def test_contains_returns_python_bool():
    G = groebner_basis([x**2 + y, x * y - 1], x, y, order='degrevlex')
    assert isinstance(G.contains(0), bool)
    assert isinstance(G.contains(1), bool)


def test_dunder_contains_matches_contains_method():
    G = groebner_basis([x**2 + y, x * y - 1], x, y, order='degrevlex')
    member = x * (x**2 + y) + y**2 * (x * y - 1)
    assert (member in G) == G.contains(member)
    assert (1 in G) == G.contains(1)
