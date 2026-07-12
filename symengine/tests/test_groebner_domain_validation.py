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

import threading

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


# ---------------------------------------------------------------------------
# P0-2 -- normal_form / is_groebner / is_reduced_basis / is_zero_dimensional
# used to construct GFpCoeffDomain(modulus) without validating
# is_supported_prime_modulus (prime, 2 <= p < 2^31), the same check
# groebner_basis already applies. Composite moduli and p >= 2^31 must now
# raise instead of silently computing with overflow-prone or non-field
# arithmetic.
# ---------------------------------------------------------------------------

_COMPOSITE_MODULUS = 6
_OVERSIZED_MODULUS = (1 << 62) + 1


def _small_basis():
    x, y = Symbol("x"), Symbol("y")
    return [x**2 - y, y**2 - x], [x, y]


@pytest.mark.parametrize("bad_modulus", [_COMPOSITE_MODULUS, _OVERSIZED_MODULUS])
def test_normal_form_rejects_bad_modulus(bad_modulus):
    basis, gens = _small_basis()
    x = gens[0]
    with pytest.raises(RuntimeError, match="Unsupported coefficient domain"):
        normal_form(x**2, basis, gens, modulus=bad_modulus)


@pytest.mark.parametrize("bad_modulus", [_COMPOSITE_MODULUS, _OVERSIZED_MODULUS])
def test_is_groebner_rejects_bad_modulus(bad_modulus):
    basis, gens = _small_basis()
    with pytest.raises(RuntimeError, match="Unsupported coefficient domain"):
        is_groebner(basis, gens, modulus=bad_modulus)


@pytest.mark.parametrize("bad_modulus", [_COMPOSITE_MODULUS, _OVERSIZED_MODULUS])
def test_is_reduced_basis_rejects_bad_modulus(bad_modulus):
    basis, gens = _small_basis()
    with pytest.raises(RuntimeError, match="Unsupported coefficient domain"):
        is_reduced_basis(basis, gens, modulus=bad_modulus)


@pytest.mark.parametrize("bad_modulus", [_COMPOSITE_MODULUS, _OVERSIZED_MODULUS])
def test_is_zero_dimensional_rejects_bad_modulus(bad_modulus):
    basis, gens = _small_basis()
    with pytest.raises(RuntimeError, match="Unsupported coefficient domain"):
        is_zero_dimensional(basis, gens, modulus=bad_modulus)


def test_auxiliary_predicates_accept_genuine_prime_modulus():
    """Sanity: a genuine prime modulus is unaffected by the new checks."""
    basis, gens = _small_basis()
    x = gens[0]
    assert normal_form(x**2, basis, gens, modulus=5) is not None
    assert is_groebner(basis, gens, modulus=5) is True
    assert is_reduced_basis(basis, gens, modulus=5) in (True, False)
    assert is_zero_dimensional(basis, gens, modulus=5) is True


# ---------------------------------------------------------------------------
# P0-1 -- check_cancellation compares the legacy max_milliseconds option
# against the thread-local active_groebner_start, which used to be set only
# by groebner_basis. On a fresh thread that never ran groebner_basis first,
# standalone normal_form / is_groebner / is_reduced_basis / is_zero_dimensional
# calls with max_milliseconds > 0 compared against the default-constructed
# (epoch) time point and failed instantly. Each entry point must now set up
# its own timer, so a generous max_milliseconds succeeds even as the very
# first Groebner-adjacent call on a brand new thread.
# ---------------------------------------------------------------------------


def _run_in_fresh_thread(fn):
    """Runs fn() on a brand new thread and returns (result, exception)."""
    result = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - propagate to caller
            result["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _non_coprime_basis():
    """A Groebner basis whose leading monomials pairwise share a variable.

    Unlike _small_basis() ({x**2 - y, y**2 - x}, whose two leading
    monomials are coprime), is_groebner_dispatch's product-criterion
    shortcut cannot skip any pair here, so the pairwise check actually
    calls normal_form_impl (and therefore check_cancellation) instead of
    trivially short-circuiting -- making it a meaningful witness for the
    max_milliseconds thread-local regression.
    """
    x, y = Symbol("x"), Symbol("y")
    return [x**2 - y, x * y - 1, y**2 - x], [x, y]


def test_normal_form_max_milliseconds_on_fresh_thread():
    basis, gens = _small_basis()
    x = gens[0]

    def call():
        return normal_form(x**2, basis, gens, max_milliseconds=60000)

    assert _run_in_fresh_thread(call) is not None


def test_is_groebner_max_milliseconds_on_fresh_thread():
    basis, gens = _non_coprime_basis()

    def call():
        return is_groebner(basis, gens, order="grlex", max_milliseconds=60000)

    assert _run_in_fresh_thread(call) is True


def test_is_reduced_basis_max_milliseconds_on_fresh_thread():
    basis, gens = _non_coprime_basis()

    def call():
        return is_reduced_basis(basis, gens, order="grlex", max_milliseconds=60000)

    # Either verdict is a legitimate reduced-basis answer; the point of this
    # regression is that the call completes rather than raising a spurious
    # resource-limit error on a fresh thread.
    assert _run_in_fresh_thread(call) in (True, False)
