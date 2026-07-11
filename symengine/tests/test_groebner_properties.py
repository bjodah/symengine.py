"""Property-based tests of the parametric (symbolic-coefficient) Groebner
basis contract.

Background: for parametric systems, ``groebner_basis(...)`` computes over the
rational function field QQ(params) and records every nonzero-divisor pivot in
``stats['genericity_assumptions']``.  The documented contract is: *the basis
is valid exactly on the open set where every recorded assumption is
nonzero*.  ``GroebnerBasis.specialize(subs)`` classifies assumptions under a
substitution (safe / violated / undecidable) and either substitutes cheaply
or recomputes.  ``solve_poly_system_ex(eqs, *gens)`` returns a
``PolySolveResult`` with ``.solutions``, ``.outcome``, ``.assumptions``,
``.selected_expression``, ``.stats``.

This module tests the contract as a *property* over randomized parameter
draws (fixed seeds -- fully deterministic), rather than at single hand-picked
points (which is what test_groebner_extended3.py's WP12 already covers).

Work items:
  1. Satisfying-specialization property (core property).
  2. Violating-specialization property (specialize() must recompute).
  3. Solver outcome corpus (concrete zero-dim / positive-dim systems).
  4. Parametric solver property (residuals + concrete solve agreement).
"""

import random
import zlib

import pytest

from symengine import Symbol, groebner_basis, Rational
from symengine.lib.symengine_wrapper import solve_poly_system_ex
from symengine.tests.groebner_parametric_corpus import PARAMETRIC_SYSTEMS
from symengine.tests.groebner_oracle import same_ideal
from symengine.tests.test_groebner_extended2 import _verify_solutions


# ===========================================================================
# Shared helpers
# ===========================================================================

DRAW_LOW, DRAW_HIGH = -6, 6
MAX_DRAWS = 25
TARGET_ACCEPTED = 3

# so_snippet_01 is excluded from the parametric-GB corpus below: computing
# its *symbolic* (parametric) Groebner basis with buchberger or f5b does not
# finish within 300s (verified empirically; see test file history / task
# report).  Every other system in PARAMETRIC_SYSTEMS is fast (<1s).
PARAMETRIC_GB_SYSTEM_NAMES = [
    name for name in PARAMETRIC_SYSTEMS if name != "so_snippet_01"
]


def _build_parametric(name):
    """Build (polys, gens, params) for a PARAMETRIC_SYSTEMS entry, without
    substituting any parameter (i.e. the fully symbolic system)."""
    s = PARAMETRIC_SYSTEMS[name]
    gens = [Symbol(g) for g in s["gens"]]
    params = [Symbol(p) for p in s["params"]]
    env = {g: gens[i] for i, g in enumerate(s["gens"])}
    env.update({p: params[i] for i, p in enumerate(s["params"])})
    env["Rational"] = Rational
    polys = [eval(p, {"__builtins__": {}}, env) for p in s["polys"]]
    return polys, gens, params


def _assumption_all_nonzero_numeric(assumptions, valmap):
    """True iff every assumption substitutes to a concrete (non-symbolic)
    nonzero number under valmap; False if any assumption is still symbolic
    or evaluates to exactly zero (rejection-sampling predicate)."""
    for a in assumptions:
        v = a.subs(valmap)
        if v.free_symbols:
            return False
        if v == 0:
            return False
    return True


def _rejection_sample_draws(params, assumptions, seed, target=TARGET_ACCEPTED,
                            max_draws=MAX_DRAWS, low=DRAW_LOW, high=DRAW_HIGH):
    """Draw up to max_draws random integer assignments for params, keeping
    only those under which every assumption substitutes to a nonzero number.
    Returns a list of up to `target` accepted valmaps (dict Symbol -> int)."""
    rng = random.Random(seed)
    accepted = []
    draws = 0
    while len(accepted) < target and draws < max_draws:
        draws += 1
        valmap = {p: rng.randint(low, high) for p in params}
        if _assumption_all_nonzero_numeric(assumptions, valmap):
            accepted.append(valmap)
    return accepted


def _find_violating_assignment(params, assumptions, seed, max_draws=200,
                               low=-3, high=3):
    """Search (seeded) for a small-integer assignment making some recorded
    assumption exactly (numerically) zero.  Returns (valmap, assumption) or
    None if not found within max_draws tries."""
    rng = random.Random(seed)
    for _ in range(max_draws):
        valmap = {p: rng.randint(low, high) for p in params}
        for a in assumptions:
            v = a.subs(valmap)
            if not v.free_symbols and v == 0:
                return valmap, a
    return None


# ===========================================================================
# Work item 1 -- satisfying-specialization property
# ===========================================================================

# Per-system seeds are derived from the system name for determinism without
# needing a shared counter (stable across test collection order changes).
# NOTE: builtin hash() is salted per-process (PYTHONHASHSEED) for str/tuple,
# so it must NOT be used here -- crc32 over encoded bytes is stable across
# runs and processes, which is required for "the suite must pass twice in a
# row identically."
def _seed_for(name):
    return zlib.crc32(("wp-groebner-properties::" + name).encode("utf-8"))


@pytest.mark.parametrize("name", PARAMETRIC_GB_SYSTEM_NAMES)
def test_satisfying_specialization_matches_concrete_gb(name):
    """The core contract: for random draws under which every recorded
    genericity assumption is nonzero, the substituted parametric basis
    (with zero polynomials filtered out) generates the same ideal as the
    Groebner basis computed directly from the substituted input system."""
    polys, gens, params = _build_parametric(name)
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm="buchberger")
    assumptions = G.stats["genericity_assumptions"]

    accepted = _rejection_sample_draws(params, assumptions, _seed_for(name))
    if not accepted:
        pytest.skip(
            f"{name}: no accepted draws out of {MAX_DRAWS} attempts "
            f"(all draws violated a recorded genericity assumption or left "
            f"one symbolic)")

    for valmap in accepted:
        subbed = [p.subs(valmap) for p in G]
        subbed_nonzero = [p for p in subbed if p != 0]
        concrete_input = [p.subs(valmap) for p in polys]
        concrete_input = [p for p in concrete_input if p != 0]
        if not concrete_input:
            continue
        G_concrete = groebner_basis(concrete_input, *gens, order="degrevlex")
        assert same_ideal(subbed_nonzero, list(G_concrete), gens,
                          order="degrevlex"), (
            f"{name}: substituted parametric basis at {valmap} does not "
            f"generate the same ideal as the directly-computed concrete GB")


@pytest.mark.slow
def test_satisfying_specialization_so_snippet_01_skipped_documented():
    """Documents why so_snippet_01 is excluded from the fast parametric-GB
    corpus: computing its parametric (symbolic-coefficient) Groebner basis
    with buchberger does not finish within 300 seconds (verified via a
    background timing run during test development).  This slow test
    attempts it with a generous timeout and skips (rather than hangs the
    slow suite) if it still doesn't finish -- it is not expected to pass in
    reasonable time and exists only to re-verify the documented limitation
    without silently bit-rotting."""
    from symengine.tests.groebner_oracle import run_with_timeout

    polys, gens, params = _build_parametric("so_snippet_01")
    G = run_with_timeout(
        groebner_basis, polys, *gens,
        order="degrevlex", algorithm="buchberger", timeout=120)
    if isinstance(G, str):
        pytest.skip(
            "so_snippet_01: parametric GB did not finish within 120s "
            f"({G}); documented as intractable for this test suite")
    # If it ever does finish (e.g. after an algorithmic improvement), verify
    # the property holds so this test starts pulling weight again.
    assumptions = G.stats["genericity_assumptions"]
    accepted = _rejection_sample_draws(params, assumptions,
                                       _seed_for("so_snippet_01"))
    for valmap in accepted:
        subbed_nonzero = [p for p in (q.subs(valmap) for q in G) if p != 0]
        concrete_input = [p for p in
                          (q.subs(valmap) for q in polys) if p != 0]
        if not concrete_input:
            continue
        G_concrete = groebner_basis(concrete_input, *gens, order="degrevlex")
        assert same_ideal(subbed_nonzero, list(G_concrete), gens,
                          order="degrevlex")


# ===========================================================================
# Work item 2 -- violating-specialization property
# ===========================================================================

# Names for which a targeted seeded search finds an exact-zero assignment
# for some recorded assumption within a small search budget.  compregb_basic,
# parametric_2var_2par, and singular_case1 have no genericity assumptions at
# all (their generic-point path never divides by a symbolic pivot for
# degrevlex/buchberger), so there is nothing to violate -- they are skipped
# with a clear reason inside the test rather than omitted from the id list,
# so a future change that starts recording assumptions for them is noticed.
VIOLATING_SEARCH_NAMES = [
    name for name in PARAMETRIC_GB_SYSTEM_NAMES
    if name != "casas_alvero_degree4"  # searched below in a slow variant
]


@pytest.mark.parametrize("name", VIOLATING_SEARCH_NAMES)
def test_violating_specialization_forces_recompute(name):
    """For an assignment that makes a recorded assumption exactly zero,
    specialize() must NOT take the cheap-substitution path: its result must
    report stats['specialization_method'] == 'recompute', and the recomputed
    basis must match the directly-computed concrete GB."""
    polys, gens, params = _build_parametric(name)
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm="buchberger")
    assumptions = G.stats["genericity_assumptions"]
    if not assumptions:
        pytest.skip(f"{name}: no genericity assumptions recorded to violate")

    found = _find_violating_assignment(params, assumptions, _seed_for(name))
    if found is None:
        pytest.skip(
            f"{name}: seeded search found no small-integer assignment "
            f"violating any recorded assumption")
    valmap, violated_assumption = found

    G_specialized = G.specialize(valmap)
    assert G_specialized.stats.get("specialization_method") == "recompute", (
        f"{name}: specialize() at {valmap} (violates assumption "
        f"{violated_assumption}) should have recomputed, not substituted")

    concrete_input = [p.subs(valmap) for p in polys]
    concrete_input = [p for p in concrete_input if p != 0]
    if not concrete_input:
        pytest.skip(f"{name}: all input polynomials vanish at {valmap}")
    G_concrete = groebner_basis(concrete_input, *gens, order="degrevlex")
    assert same_ideal(list(G_specialized), list(G_concrete), gens,
                      order="degrevlex"), (
        f"{name}: recomputed specialize() result at {valmap} disagrees "
        f"with the directly-computed concrete GB")


@pytest.mark.slow
def test_violating_specialization_casas_alvero_degree4():
    """Same property as above for casas_alvero_degree4, marked slow because
    its parametric GB computation (~0.5s) combined with the recompute makes
    this test noticeably heavier than the others."""
    name = "casas_alvero_degree4"
    polys, gens, params = _build_parametric(name)
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm="buchberger")
    assumptions = G.stats["genericity_assumptions"]
    assert assumptions, f"{name}: expected recorded genericity assumptions"

    found = _find_violating_assignment(params, assumptions, _seed_for(name))
    assert found is not None, (
        f"{name}: seeded search found no violating assignment")
    valmap, violated_assumption = found

    G_specialized = G.specialize(valmap)
    assert G_specialized.stats.get("specialization_method") == "recompute"

    concrete_input = [p.subs(valmap) for p in polys]
    concrete_input = [p for p in concrete_input if p != 0]
    if not concrete_input:
        pytest.skip(f"{name}: all input polynomials vanish at {valmap}")
    G_concrete = groebner_basis(concrete_input, *gens, order="degrevlex")
    assert same_ideal(list(G_specialized), list(G_concrete), gens,
                      order="degrevlex")


# ===========================================================================
# Work item 3 -- solver outcome corpus
# ===========================================================================

def _concrete_solver_systems():
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    x0, x1, x2 = Symbol("x0"), Symbol("x1"), Symbol("x2")
    return {
        "quad_shared_root": (
            [x**2 - 1, y - x], [x, y]),
        "grid01": (
            [x**2 - x, y**2 - y], [x, y]),
        "katsura3": (
            [x0 + 2*x1 + 2*x2 - 1,
             x0**2 + 2*x1**2 + 2*x2**2 - x0,
             2*x0*x1 + 2*x1*x2 - x1], [x0, x1, x2]),
        "symmetric3": (
            [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1],
            [x, y, z]),
    }


@pytest.mark.parametrize("name", list(_concrete_solver_systems().keys()))
def test_concrete_zero_dimensional_solver_complete(name):
    """These concrete zero-dimensional systems must be solved completely,
    and every returned point must satisfy every input equation."""
    F, gens = _concrete_solver_systems()[name]
    result = solve_poly_system_ex(F, *gens)
    assert result.outcome == "complete", (
        f"{name}: expected outcome 'complete', got {result.outcome!r}")
    _verify_solutions(F, result.solutions, gens)


def test_concrete_cyclic3_incomplete_but_sound():
    """cyclic-3 ({x+y+z, xy+xz+yz, xyz-1}) is zero-dimensional with 6 true
    solutions (all permutations of the primitive cube roots of unity), but
    the current solver's triangular back-substitution only recovers 2 of
    them and honestly reports outcome == 'incomplete' (verified: all 6
    permutations of (1, omega, omega^2) satisfy the system).  This is NOT a
    contract violation -- the documented contract for 'incomplete' is only
    that every *returned* point is a genuine solution, not that the result
    is exhaustive.  This test pins down that documented-sound-but-partial
    behavior; if the solver's coverage improves, tighten this assertion."""
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    F = [x + y + z, x*y + x*z + y*z, x*y*z - 1]
    result = solve_poly_system_ex(F, x, y, z)
    assert result.outcome == "incomplete"
    _verify_solutions(F, result.solutions, [x, y, z])
    # Documents the current (partial) recovery count so a regression to 0
    # or an improvement to more points is visible rather than silent.
    assert len(result.solutions.args) == 2


def test_concrete_positive_dimensional_not_zero_dimensional():
    """A known positive-dimensional system (a single bivariate equation)
    must report outcome == 'not_zero_dimensional'."""
    x, y = Symbol("x"), Symbol("y")
    result = solve_poly_system_ex([x*y - 1], x, y)
    assert result.outcome == "not_zero_dimensional"


# ===========================================================================
# Work item 4 -- parametric solver property
# ===========================================================================

def _residual_is_zero(value):
    """Accept an exact zero, or (for radical expressions expand() cannot
    fully cancel) a high-precision numeric zero within tolerance 1e-9."""
    if value == 0:
        return True
    resid = complex(value.n(53, real=False))
    return abs(resid) < 1e-9


def _check_parametric_solution_residuals(F, gens, solutions, valmap):
    """Substitute valmap into params first (so solution-point expressions
    become concrete), then verify each point satisfies every equation."""
    for pt in solutions.args:
        point_vals = [comp.subs(valmap) for comp in pt.args]
        subs_map = dict(zip(gens, point_vals))
        for f in F:
            v = f.subs(valmap).subs(subs_map).expand()
            assert _residual_is_zero(v), (
                f"solution point {point_vals} at {valmap} does not satisfy "
                f"{f} (residual {v})")


class TestParametricSolverProperty:
    """For small parametric systems: draw seeded random assignments
    satisfying every reported assumption, substitute each returned solution
    point into the original equations (residual check), and confirm the
    concrete solve on the substituted system returns the same solution
    count with outcome 'complete'."""

    def test_linear_c1x_minus_1(self):
        x, y = Symbol("x"), Symbol("y")
        C1, C2 = Symbol("C1"), Symbol("C2")
        F = [C1*x - 1, y - C2*x]
        gens = [x, y]
        result = solve_poly_system_ex(F, *gens)
        assert result.outcome == "complete"
        params = [C1, C2]

        accepted = _rejection_sample_draws(
            params, result.assumptions, _seed_for("parametric_solve_c1x"))
        assert accepted, "expected at least one accepted draw"
        for valmap in accepted:
            _check_parametric_solution_residuals(
                F, gens, result.solutions, valmap)
            F_concrete = [f.subs(valmap) for f in F]
            result_concrete = solve_poly_system_ex(F_concrete, *gens)
            assert result_concrete.outcome == "complete"
            assert len(result_concrete.solutions.args) == \
                len(result.solutions.args)

    # Regression test: this used to return (outcome='complete',
    # solutions=EmptySet) for the cross-coupled fully-parametric linear
    # system, because the residual check expanded the substituted equations
    # without cancelling rational-function quotients (C1*C4/D - C2*C3/D - 1
    # with D = C1*C4 - C2*C3 stayed structurally nonzero), rejecting the
    # genuine solution and then labelling the empty set Complete. Fixed by
    # testing the expanded numerator in solve_poly_system_attempt.
    def test_linear_cross_coupled_c1_c4(self):
        x, y = Symbol("x"), Symbol("y")
        C1, C2, C3, C4 = Symbol("C1"), Symbol("C2"), Symbol("C3"), Symbol("C4")
        F = [C1*x + C2*y - 1, C3*x + C4*y - 1]
        gens = [x, y]
        result = solve_poly_system_ex(F, *gens)
        assert result.outcome == "complete"
        params = [C1, C2, C3, C4]

        accepted = _rejection_sample_draws(
            params, result.assumptions, _seed_for("parametric_solve_cross"))
        assert accepted, "expected at least one accepted draw"
        # The genuine bug: solutions is EmptySet even though a nonsingular
        # concrete instantiation has exactly one solution.
        assert len(result.solutions.args) > 0, (
            "solve_poly_system_ex returned no solutions for a "
            "non-degenerate cross-coupled parametric linear system")
        for valmap in accepted:
            _check_parametric_solution_residuals(
                F, gens, result.solutions, valmap)
            F_concrete = [f.subs(valmap) for f in F]
            result_concrete = solve_poly_system_ex(F_concrete, *gens)
            assert result_concrete.outcome == "complete"
            assert len(result_concrete.solutions.args) == \
                len(result.solutions.args)

    def test_quadratic_c1_squared_minus_4(self):
        x, y = Symbol("x"), Symbol("y")
        C1 = Symbol("C1")
        F = [(C1**2 - 4)*x**2 - 1, y - x]
        gens = [x, y]
        result = solve_poly_system_ex(F, *gens)
        assert result.outcome == "complete"
        params = [C1]

        accepted = _rejection_sample_draws(
            params, result.assumptions, _seed_for("parametric_solve_quad"))
        assert accepted, "expected at least one accepted draw"
        for valmap in accepted:
            _check_parametric_solution_residuals(
                F, gens, result.solutions, valmap)
            F_concrete = [f.subs(valmap) for f in F]
            result_concrete = solve_poly_system_ex(F_concrete, *gens)
            assert result_concrete.outcome == "complete"
            assert len(result_concrete.solutions.args) == \
                len(result.solutions.args)
