"""P2-1: unknown Groebner kwargs used to be silently ignored.

_build_groebner_options[_with_holder] (symengine_wrapper.in.pyx) reads known
keys with .get() and, before this fix, never rejected leftovers: a typo like
algoritm='f5b' silently computed with the default (Auto) algorithm instead of
raising. This closes that gap for every public **kwargs entry point that
routes through it: groebner_basis, normal_form, is_groebner,
is_reduced_basis, is_zero_dimensional, solve_poly_system,
solve_poly_system_ex.

Also documents the current full recognized-key surface, including the P2-2 /
P2-5 additions (limits, cancellation_token, normalize).
"""

import pytest

from symengine import Symbol, groebner_basis
from symengine.lib.symengine_wrapper import (
    normal_form, is_groebner, is_reduced_basis, is_zero_dimensional,
    solve_poly_system, solve_poly_system_ex,
)

x, y = Symbol('x'), Symbol('y')


def _basis():
    return [x**2 - y, y**2 - x]


def test_groebner_basis_rejects_unknown_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        groebner_basis(_basis(), x, y, algoritm='f5b')


def test_groebner_basis_rejects_unknown_kwarg_names_the_offender():
    with pytest.raises(TypeError, match="algoritm"):
        groebner_basis(_basis(), x, y, algoritm='f5b')


def test_groebner_basis_rejects_multiple_unknown_kwargs():
    with pytest.raises(TypeError) as excinfo:
        groebner_basis(_basis(), x, y, foo=1, bar=2)
    msg = str(excinfo.value)
    assert 'foo' in msg and 'bar' in msg


def test_normal_form_rejects_unknown_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        normal_form(x**2, _basis(), [x, y], oder='lex')


def test_is_groebner_rejects_unknown_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        is_groebner(_basis(), [x, y], modulos=7)


def test_is_reduced_basis_rejects_unknown_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        is_reduced_basis(_basis(), [x, y], modulos=7)


def test_is_zero_dimensional_rejects_unknown_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        is_zero_dimensional(_basis(), [x, y], odrer='lex')


def test_solve_poly_system_rejects_unknown_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        solve_poly_system(_basis(), x, y, algoritm='f5b')


def test_solve_poly_system_ex_rejects_unknown_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        solve_poly_system_ex(_basis(), x, y, algoritm='f5b')


# ---------------------------------------------------------------------------
# Sanity: every currently-recognized keyword still works (nothing here was
# accidentally dropped from the known-keys set while adding the check).
# ---------------------------------------------------------------------------

def test_all_recognized_kwargs_accepted():
    G = groebner_basis(
        _basis(), x, y,
        order='degrevlex',
        algorithm='auto',
        reduced=True,
        interreduce_input=True,
        sort_output=True,
        modulus=0,
        max_s_pairs=0,
        max_reduction_steps=0,
        max_milliseconds=0,
        max_degree=0,
        max_coefficient_ops=0,
        track_genericity=True,
        cancellation_check_interval=1024,
        normalize='monic',
        cancellation_token=None,
        limits=None,
    )
    assert len(G.polys) > 0


# ---------------------------------------------------------------------------
# Internal call sites (GroebnerBasis.is_zero_dimensional / .fglm / .reduce)
# build their own small kwargs dicts and route through the same validation;
# they must keep working (they only ever pass a known-safe subset).
# ---------------------------------------------------------------------------

def test_internal_call_sites_still_work():
    G = groebner_basis(_basis(), x, y, order='degrevlex')
    assert G.is_zero_dimensional is True
    assert G.reduce(x**3) is not None
    G_lex = G.fglm('lex')
    assert len(G_lex.polys) > 0
