"""Tests closing the domain-validation gap in normal_form, is_groebner,
is_reduced_basis, and is_zero_dimensional.

groebner_basis validates that symbolic coefficients are rational functions
in QQ(parameters) before running the structural ExpressionCoeffDomain
arithmetic (that arithmetic relies on exact structural zero tests, which are
unsound for something like sin(t)**2 + cos(t)**2 - 1). The other four public
entry points took the same symbolic dispatch path without the same check;
they now raise on an unsupported coefficient domain instead of silently
running unsound arithmetic.

The C++ SymEngineException surfaces through Cython's default `except +`
translation as a plain RuntimeError (there is no custom exception
translator registered for it), matching how groebner_basis's own
UnsupportedCoefficientDomain status is worded ("Unsupported coefficient
domain").
"""

import pytest

from symengine import Symbol, sin, groebner_basis
from symengine.lib.symengine_wrapper import (
    normal_form, is_groebner, is_reduced_basis, is_zero_dimensional,
)


def _sin_coefficient_basis():
    t, x = Symbol("t"), Symbol("x")
    return [sin(t) * x + 1], [x]


def test_normal_form_rejects_sin_coefficient():
    bad_basis, gens = _sin_coefficient_basis()
    x = gens[0]
    with pytest.raises(RuntimeError, match="Unsupported coefficient domain"):
        normal_form(x**2, bad_basis, gens)


def test_is_groebner_rejects_sin_coefficient():
    bad_basis, gens = _sin_coefficient_basis()
    with pytest.raises(RuntimeError, match="Unsupported coefficient domain"):
        is_groebner(bad_basis, gens)


def test_is_reduced_basis_rejects_sin_coefficient():
    bad_basis, gens = _sin_coefficient_basis()
    with pytest.raises(RuntimeError, match="Unsupported coefficient domain"):
        is_reduced_basis(bad_basis, gens)


def test_is_zero_dimensional_rejects_sin_coefficient():
    bad_basis, gens = _sin_coefficient_basis()
    with pytest.raises(RuntimeError, match="Unsupported coefficient domain"):
        is_zero_dimensional(bad_basis, gens)


def test_rational_function_parameters_still_work():
    """Sanity: genuinely rational-function-in-parameters symbolic bases
    (the supported case) are unaffected by the new check."""
    x = Symbol("x")
    c = Symbol("c")
    good_basis = [c * x - 1]
    gens = [x]
    assert normal_form(x**2, good_basis, gens) is not None
    assert is_groebner(good_basis, gens) is True


def test_groebner_basis_itself_already_rejects_sin_coefficient():
    """groebner_basis already validated this case; kept here so the four
    newly-validated entry points can be read against a known-good
    reference."""
    t, x = Symbol("t"), Symbol("x")
    with pytest.raises(ValueError, match="Unsupported coefficient domain"):
        groebner_basis([sin(t) * x + 1], x)
