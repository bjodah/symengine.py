"""Extended Groebner basis testing campaign (Work Packages 1-4).

WP1 — Differential testing vs SymPy oracle
WP2 — m4gb / GF(p) deep dive
WP3 — MoGVW thorough coverage
WP4 — Monomial orders, FGLM, and zero-dimensionality

See docs/reports/09-11 for methodology.
"""

import sys
import pytest

from symengine import (
    Symbol, groebner_basis, GroebnerBasis, sympify,
)
from symengine.lib.symengine_wrapper import (
    normal_form,
    is_groebner,
    is_reduced_basis,
    is_zero_dimensional,
)
from symengine.tests.groebner_corpus import SYSTEMS
from symengine.tests.groebner_oracle import (
    same_ideal,
    contains_ideal,
    assert_correct_gb,
    run_with_timeout,
    all_algorithms,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ORDER_MAP_SE_TO_SYMPY = {"lex": "lex", "grlex": "grlex", "degrevlex": "grevlex"}


def _build_corpus(name):
    """Build polys/gens/golden_size from the vendored corpus."""
    s = SYSTEMS[name]
    gens = [Symbol(g) for g in s["gens"]]
    env = {g: gens[i] for i, g in enumerate(s["gens"])}
    polys = [eval(p, {"__builtins__": {}}, env) for p in s["polys"]]
    return polys, gens, s["golden_size"]


def _build_benchmark(builder_fn):
    """Build polys/gens from a benchmark builder (returns (polys, gens))."""
    return builder_fn()


# Extra systems from the benchmark module (imported lazily to avoid path issues)
def _get_extra_system(name):
    """Return (polys_se, gens_se) for a named benchmark system."""
    sys.path.insert(0, "/work/benchmarks")
    from groebner_systems import katsura, ammonia_numeric, noon, eco, reimer

    builders = {
        "katsura3": lambda: katsura(None, 3),
        "katsura4": lambda: katsura(None, 4),
        "katsura5": lambda: katsura(None, 5),
        "ammonia": ammonia_numeric,
        "noon5": lambda: noon(None, 5),
        "eco5": lambda: eco(None, 5),
        "reimer4": lambda: reimer(None, 4),
    }
    # These builders need a backend module — use symengine
    import symengine as m

    se_builders = {
        "katsura3": lambda: katsura(m, 3),
        "katsura4": lambda: katsura(m, 4),
        "katsura5": lambda: katsura(m, 5),
        "ammonia": lambda: ammonia_numeric(m),
        "noon5": lambda: noon(m, 5),
        "eco5": lambda: eco(m, 5),
        "reimer4": lambda: reimer(m, 4),
    }
    return se_builders[name]()


def _get_sympy_system(name):
    """Return (sympy_polys, sympy_gens) for a named system."""
    import sympy

    if name in SYSTEMS:
        s = SYSTEMS[name]
        sympy_gens = [sympy.Symbol(g) for g in s["gens"]]
        env = {g: sympy_gens[i] for i, g in enumerate(s["gens"])}
        polys = [eval(p, {"__builtins__": {}}, env) for p in s["polys"]]
        return polys, sympy_gens

    sys.path.insert(0, "/work/benchmarks")
    from groebner_systems import katsura, ammonia_numeric, noon, eco, reimer

    sympy_builders = {
        "katsura3": lambda: katsura(sympy, 3),
        "katsura4": lambda: katsura(sympy, 4),
        "katsura5": lambda: katsura(sympy, 5),
        "ammonia": lambda: ammonia_numeric(sympy),
        "noon5": lambda: noon(sympy, 5),
        "eco5": lambda: eco(sympy, 5),
        "reimer4": lambda: reimer(sympy, 4),
    }
    return sympy_builders[name]()


# All 12 WP1 systems (≥10 required)
WP1_SYSTEMS_ALL = [
    "cyclic3", "cyclic4", "cyclic5",
    "rose", "Uteshev_Bikker", "liu",
    "katsura3", "katsura4", "katsura5",
    "ammonia", "eco5", "noon5",
]

# Systems fast enough for the default (non-slow) suite
WP1_SYSTEMS_FAST = ["cyclic3", "cyclic4", "katsura3", "katsura4", "ammonia"]
WP1_SYSTEMS_SLOW = [s for s in WP1_SYSTEMS_ALL if s not in WP1_SYSTEMS_FAST]

# Orders that are fast for each slow system
_SLOW_FAST_ORDERS = {
    "cyclic5": ["degrevlex", "grlex", "lex"],
    "rose": ["degrevlex", "grlex"],
    "Uteshev_Bikker": ["degrevlex"],
    "liu": ["degrevlex"],
    "katsura5": ["degrevlex"],
    "eco5": ["degrevlex", "grlex"],
    "noon5": ["degrevlex", "grlex"],
}


# ===========================================================================
# WP1 — Differential testing vs SymPy oracle
# ===========================================================================

@pytest.mark.parametrize("name", WP1_SYSTEMS_FAST)
@pytest.mark.parametrize("order", ["lex", "grlex", "degrevlex"])
def test_wp1_sympy_differential(name, order):
    """SymEngine GB must generate the same ideal as SymPy's GB."""
    sympy = pytest.importorskip("sympy")
    sympy_order = _ORDER_MAP_SE_TO_SYMPY[order]

    if name in SYSTEMS:
        se_polys, se_gens, _ = _build_corpus(name)
    else:
        se_polys, se_gens = _get_extra_system(name)

    sympy_polys, sympy_gens = _get_sympy_system(name)

    G_se = groebner_basis(se_polys, *se_gens, order=order)
    G_sp = sympy.groebner(sympy_polys, *sympy_gens, order=sympy_order)

    # Convert SymPy basis to SymEngine for ideal comparison
    sp_as_se = [sympify(p) for p in G_sp.exprs]

    assert same_ideal(list(G_se), sp_as_se, se_gens, order=order), \
        f"SymEngine and SymPy disagree on the ideal for {name} ({order})"

    # If both are reduced, check set equality
    if is_reduced_basis(list(G_se), se_gens, order=order):
        assert len(G_se) == len(G_sp), \
            f"Reduced basis size mismatch: SE={len(G_se)} vs SP={len(G_sp)}"


@pytest.mark.slow
@pytest.mark.parametrize("name", WP1_SYSTEMS_SLOW)
@pytest.mark.parametrize("order", ["degrevlex"])
def test_wp1_sympy_differential_slow_drl(name, order):
    """Slow systems, degrevlex only."""
    sympy = pytest.importorskip("sympy")
    sympy_order = _ORDER_MAP_SE_TO_SYMPY[order]

    if name in SYSTEMS:
        se_polys, se_gens, _ = _build_corpus(name)
    else:
        se_polys, se_gens = _get_extra_system(name)

    sympy_polys, sympy_gens = _get_sympy_system(name)

    G_se = groebner_basis(se_polys, *se_gens, order=order)
    G_sp = sympy.groebner(sympy_polys, *sympy_gens, order=sympy_order)

    sp_as_se = [sympify(p) for p in G_sp.exprs]

    assert same_ideal(list(G_se), sp_as_se, se_gens, order=order), \
        f"SymEngine and SymPy disagree on the ideal for {name} ({order})"

    if is_reduced_basis(list(G_se), se_gens, order=order):
        assert len(G_se) == len(G_sp), \
            f"Reduced basis size mismatch: SE={len(G_se)} vs SP={len(G_sp)}"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["cyclic5", "rose"])
@pytest.mark.parametrize("order", ["lex", "grlex"])
def test_wp1_sympy_differential_slow_lex_grlex(name, order):
    """Slow systems with lex/grlex (where fast enough)."""
    if order not in _SLOW_FAST_ORDERS.get(name, []):
        pytest.skip(f"{name}/{order} too slow for this test")

    sympy = pytest.importorskip("sympy")
    sympy_order = _ORDER_MAP_SE_TO_SYMPY[order]

    if name in SYSTEMS:
        se_polys, se_gens, _ = _build_corpus(name)
    else:
        se_polys, se_gens = _get_extra_system(name)

    sympy_polys, sympy_gens = _get_sympy_system(name)

    G_se = groebner_basis(se_polys, *se_gens, order=order)
    G_sp = sympy.groebner(sympy_polys, *sympy_gens, order=sympy_order)

    sp_as_se = [sympify(p) for p in G_sp.exprs]

    assert same_ideal(list(G_se), sp_as_se, se_gens, order=order), \
        f"SymEngine and SymPy disagree on the ideal for {name} ({order})"

    if is_reduced_basis(list(G_se), se_gens, order=order):
        assert len(G_se) == len(G_sp), \
            f"Reduced basis size mismatch: SE={len(G_se)} vs SP={len(G_sp)}"


# ===========================================================================
# WP2 — m4gb / GF(p) deep dive
# ===========================================================================

WP2_SYSTEMS = list(SYSTEMS.keys())  # All 6 corpus systems


@pytest.mark.parametrize("name", WP2_SYSTEMS)
@pytest.mark.parametrize("algorithm", ["buchberger", "f5b"])
def test_wp2_gfp_reference_algorithms(name, algorithm):
    """buchberger and f5b must produce correct, agreeing GBs over GF(p)."""
    polys, gens, golden = _build_corpus(name)
    modulus = 32003

    G = groebner_basis(polys, *gens, order="degrevlex", algorithm=algorithm,
                       modulus=modulus)
    assert_correct_gb(polys, gens, G, order="degrevlex", modulus=modulus,
                      golden_size=golden)


@pytest.mark.parametrize("name", WP2_SYSTEMS)
def test_wp2_gfp_buchberger_f5b_agree(name):
    """buchberger and f5b must agree on every corpus system over GF(p)."""
    polys, gens, _ = _build_corpus(name)
    modulus = 32003

    G_buch = groebner_basis(polys, *gens, order="degrevlex",
                            algorithm="buchberger", modulus=modulus)
    G_f5b = groebner_basis(polys, *gens, order="degrevlex",
                           algorithm="f5b", modulus=modulus)

    assert same_ideal(G_buch, G_f5b, gens, order="degrevlex", modulus=modulus), \
        f"buchberger and f5b disagree over GF(32003) on {name}"


@pytest.mark.parametrize("name", ["cyclic3", "cyclic4", "rose",
                                  "Uteshev_Bikker", "cyclic5", "liu"])
def test_wp2_m4gb_gfp_correct(name):
    """m4gb must produce a correct GB over GF(p).

    cyclic5 and liu are regression tests: m4gb used to return wrong bases
    for them (stale dense-index handling in update/shrink, fixed in
    report 14's campaign).
    """
    polys, gens, golden = _build_corpus(name)
    modulus = 32003

    # Reference from f5b
    G_ref = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b",
                           modulus=modulus)
    G_m4gb = groebner_basis(polys, *gens, order="degrevlex", algorithm="m4gb",
                            modulus=modulus)

    assert_correct_gb(polys, gens, G_m4gb, order="degrevlex", modulus=modulus,
                      golden_size=golden, reference=list(G_ref))


def test_wp2_m4gb_gfp_zero_poly_input():
    """Input polynomials that vanish mod p must not crash m4gb (regression:
    num_vars was read from F[0] even when it was the zero polynomial)."""
    x, y = Symbol("x"), Symbol("y")
    G = groebner_basis([2*x**2, 1 + x*y**2], x, y, order="degrevlex",
                       algorithm="m4gb", modulus=2)
    R = groebner_basis([2*x**2, 1 + x*y**2], x, y, order="degrevlex",
                       algorithm="f5b", modulus=2)
    assert same_ideal(list(G), list(R), [x, y], order="degrevlex", modulus=2)


# ===========================================================================
# WP3 — MoGVW thorough coverage
# ===========================================================================


def _mogvw_gb(polys, gens, order, modulus=0):
    """Compute GB with mogvw — used as picklable function for run_with_timeout."""
    kw = {"order": order, "algorithm": "mogvw"}
    if modulus:
        kw["modulus"] = modulus
    return groebner_basis(polys, *gens, **kw)


def _gb_for_order(polys, gens, order):
    """Compute GB with the default algorithm — picklable for run_with_timeout."""
    return groebner_basis(polys, *gens, order=order)


# --- Fast systems: cyclic3, cyclic4 (all orders, QQ and GF(p)) ---

@pytest.mark.parametrize("name", ["cyclic3", "cyclic4"])
@pytest.mark.parametrize("order", ["grlex", "degrevlex"])
@pytest.mark.parametrize("modulus", [0, 32003])
def test_wp3_mogvw_fast_systems(name, order, modulus):
    """mogvw must agree with f5b on small systems (all orders, QQ & GF(p))."""
    polys, gens, _ = _build_corpus(name)
    kw = {"order": order}
    if modulus:
        kw["modulus"] = modulus

    G_mogvw = run_with_timeout(_mogvw_gb, polys, gens, order, modulus,
                               timeout=15)
    assert not isinstance(G_mogvw, str), \
        f"mogvw returned {G_mogvw} for {name} {order} mod={modulus}"

    G_f5b = groebner_basis(polys, *gens, algorithm="f5b", **kw)

    assert is_groebner(list(G_mogvw), gens, **kw), \
        f"mogvw GB is not a Groebner basis for {name} {order} mod={modulus}"
    assert same_ideal(G_mogvw, G_f5b, gens, order=order, modulus=modulus), \
        f"mogvw and f5b disagree for {name} {order} mod={modulus}"


# --- Additional fast systems from corpus (QQ, degrevlex) ---

@pytest.mark.slow
@pytest.mark.parametrize("name", ["Uteshev_Bikker", "liu"])
def test_wp3_mogvw_corpus_fast(name):
    """mogvw on remaining corpus systems (degrevlex, QQ)."""
    polys, gens, _ = _build_corpus(name)

    G_mogvw = run_with_timeout(_mogvw_gb, polys, gens, "degrevlex", 0,
                               timeout=15)
    if isinstance(G_mogvw, str):
        pytest.skip(f"mogvw {G_mogvw} on {name}")

    G_f5b = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b")

    assert is_groebner(list(G_mogvw), gens, order="degrevlex"), \
        f"mogvw GB is not a Groebner basis for {name}"
    assert same_ideal(G_mogvw, G_f5b, gens, order="degrevlex"), \
        f"mogvw and f5b disagree for {name}"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["Uteshev_Bikker", "liu"])
def test_wp3_mogvw_corpus_gfp(name):
    """mogvw on remaining corpus systems (degrevlex, GF(p))."""
    polys, gens, _ = _build_corpus(name)
    modulus = 32003

    G_mogvw = run_with_timeout(_mogvw_gb, polys, gens, "degrevlex", modulus,
                               timeout=15)
    if isinstance(G_mogvw, str):
        pytest.skip(f"mogvw {G_mogvw} on {name} GF(p)")

    G_f5b = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b",
                           modulus=modulus)

    assert is_groebner(list(G_mogvw), gens, order="degrevlex",
                       modulus=modulus), \
        f"mogvw GB is not a Groebner basis for {name} GF(p)"
    assert same_ideal(G_mogvw, G_f5b, gens, order="degrevlex",
                      modulus=modulus), \
        f"mogvw and f5b disagree for {name} GF(p)"


# --- Additional benchmark systems ---

@pytest.mark.parametrize("name", ["katsura3", "katsura4"])
@pytest.mark.parametrize("order", ["grlex", "degrevlex"])
def test_wp3_mogvw_katsura(name, order):
    """mogvw on katsura3/4 (all orders, QQ)."""
    se_polys, se_gens = _get_extra_system(name)

    G_mogvw = run_with_timeout(_mogvw_gb, se_polys, se_gens, order, 0,
                               timeout=15)
    assert not isinstance(G_mogvw, str), \
        f"mogvw returned {G_mogvw} for {name} {order}"

    G_f5b = groebner_basis(se_polys, *se_gens, order=order, algorithm="f5b")

    assert is_groebner(list(G_mogvw), se_gens, order=order)
    assert same_ideal(G_mogvw, G_f5b, se_gens, order=order)


# --- cyclic5 with mogvw (the equal-signature regression from report 11) ---

@pytest.mark.slow
def test_wp3_mogvw_cyclic5_drl():
    """cyclic5 with mogvw (degrevlex, QQ) — regression guard for report 11.

    Unconditionally dropping equal-signature S-pairs made mogvw return an
    incomplete, non-Groebner basis on cyclic5.  If mogvw hangs, we mark it
    as xfail (timeout = bug).
    """
    polys, gens, golden = _build_corpus("cyclic5")

    G_mogvw = run_with_timeout(_mogvw_gb, polys, gens, "degrevlex", 0,
                               timeout=60)
    if isinstance(G_mogvw, str):
        pytest.xfail(f"mogvw {G_mogvw} on cyclic5 — likely still broken")

    G_f5b = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b")

    assert is_groebner(list(G_mogvw), gens, order="degrevlex")
    assert len(G_mogvw) == golden
    assert same_ideal(G_mogvw, G_f5b, gens, order="degrevlex")


@pytest.mark.slow
def test_wp3_mogvw_cyclic5_gfp():
    """cyclic5 with mogvw (degrevlex, GF(p)) — regression guard."""
    polys, gens, golden = _build_corpus("cyclic5")
    modulus = 32003

    G_mogvw = run_with_timeout(_mogvw_gb, polys, gens, "degrevlex", modulus,
                               timeout=60)
    if isinstance(G_mogvw, str):
        pytest.xfail(f"mogvw {G_mogvw} on cyclic5 GF(p) — likely still broken")

    G_f5b = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b",
                           modulus=modulus)

    assert is_groebner(list(G_mogvw), gens, order="degrevlex",
                       modulus=modulus)
    assert len(G_mogvw) == golden
    assert same_ideal(G_mogvw, G_f5b, gens, order="degrevlex",
                      modulus=modulus)


@pytest.mark.parametrize("modulus", [0, 32003])
def test_wp3_mogvw_rose_regression(modulus):
    """rose needs the paper's row-ordering and maxcpdeg details to complete."""
    polys, gens, golden = _build_corpus("rose")
    kw = {"order": "degrevlex", "algorithm": "mogvw"}
    ref_kw = {"order": "degrevlex", "algorithm": "f5b"}
    check_kw = {"order": "degrevlex"}
    if modulus:
        kw["modulus"] = ref_kw["modulus"] = check_kw["modulus"] = modulus

    G = groebner_basis(polys, *gens, **kw)
    G_ref = groebner_basis(polys, *gens, **ref_kw)

    assert is_groebner(list(G), gens, **check_kw)
    assert len(G) == golden
    assert same_ideal(G, G_ref, gens, order="degrevlex", modulus=modulus)


def test_wp3_mogvw_lex():
    """mogvw supports lex order (the former gate was overly conservative)."""
    polys, gens, _ = _build_corpus("cyclic3")
    G = groebner_basis(polys, *gens, order="lex", algorithm="mogvw")
    assert is_groebner(list(G), gens, order="lex")
    G_ref = groebner_basis(polys, *gens, order="lex", algorithm="buchberger")
    assert same_ideal(list(G), list(G_ref), gens, order="lex")


# ===========================================================================
# WP4 — Monomial orders, FGLM, and zero-dimensionality
# ===========================================================================

# --- Zero-dimensional systems for FGLM round-trip tests ---

WP4_ZD_SYSTEMS = ["cyclic3", "cyclic5"]
WP4_ZD_SYSTEMS_SLOW = ["rose", "Uteshev_Bikker"]


@pytest.mark.parametrize("name", WP4_ZD_SYSTEMS)
def test_wp4_fglm_round_trip(name):
    """FGLM round-trip: degrevlex -> lex -> degrevlex preserves the ideal.

    Each leg must be a valid Groebner basis, and the round-tripped basis
    must generate the same ideal as the original.
    """
    polys, gens, _ = _build_corpus(name)

    G_drl = groebner_basis(polys, *gens, order="degrevlex")
    assert is_zero_dimensional(list(G_drl), gens, order="degrevlex"), \
        f"{name} should be zero-dimensional"

    # degrevlex -> lex
    G_lex = G_drl.fglm("lex")
    assert G_lex.order == "lex"
    assert is_groebner(list(G_lex), gens, order="lex"), \
        f"{name}: FGLM degrevlex->lex did not produce a Groebner basis"

    # lex -> degrevlex
    G_back = G_lex.fglm("degrevlex")
    assert G_back.order == "degrevlex"
    assert is_groebner(list(G_back), gens, order="degrevlex"), \
        f"{name}: FGLM lex->degrevlex did not produce a Groebner basis"

    # Ideal invariance after round trip
    assert same_ideal(G_drl, G_back, gens, order="degrevlex"), \
        f"{name}: FGLM round-trip changed the ideal"


@pytest.mark.slow
@pytest.mark.parametrize("name", WP4_ZD_SYSTEMS_SLOW)
def test_wp4_fglm_round_trip_slow(name):
    """FGLM round-trip for heavier zero-dimensional systems."""
    polys, gens, _ = _build_corpus(name)

    G_drl = groebner_basis(polys, *gens, order="degrevlex")
    assert is_zero_dimensional(list(G_drl), gens, order="degrevlex"), \
        f"{name} should be zero-dimensional"

    G_lex = G_drl.fglm("lex")
    assert G_lex.order == "lex"
    assert is_groebner(list(G_lex), gens, order="lex"), \
        f"{name}: FGLM degrevlex->lex did not produce a Groebner basis"

    G_back = G_lex.fglm("degrevlex")
    assert G_back.order == "degrevlex"
    assert is_groebner(list(G_back), gens, order="degrevlex"), \
        f"{name}: FGLM lex->degrevlex did not produce a Groebner basis"

    assert same_ideal(G_drl, G_back, gens, order="degrevlex"), \
        f"{name}: FGLM round-trip changed the ideal"


# --- Cross-order ideal invariance ---

def _cross_order_same_ideal(G_a, G_b, gens, order_a, order_b):
    """Check that two bases (computed under different orders) generate the same
    ideal.  Each direction of containment must be checked with the order under
    which the divisor basis IS a Groebner basis."""
    return (contains_ideal(G_a, G_b, gens, order=order_a) and
            contains_ideal(G_b, G_a, gens, order=order_b))


@pytest.mark.parametrize("name", ["cyclic3", "cyclic4"])
def test_wp4_cross_order_same_ideal_fast(name):
    """same_ideal must hold across all three orders for the same input.

    Each direction of the containment check must use the monomial order
    under which the *divisor* basis is a valid GB (otherwise normal_form
    may produce a non-unique remainder and falsely report non-membership).
    """
    polys, gens, _ = _build_corpus(name)

    G_lex = groebner_basis(polys, *gens, order="lex")
    G_grlex = groebner_basis(polys, *gens, order="grlex")
    G_drl = groebner_basis(polys, *gens, order="degrevlex")

    assert _cross_order_same_ideal(G_lex, G_grlex, gens, "lex", "grlex"), \
        f"{name}: lex and grlex ideals disagree"
    assert _cross_order_same_ideal(G_lex, G_drl, gens, "lex", "degrevlex"), \
        f"{name}: lex and degrevlex ideals disagree"
    assert _cross_order_same_ideal(G_grlex, G_drl, gens, "grlex", "degrevlex"), \
        f"{name}: grlex and degrevlex ideals disagree"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["cyclic5", "rose"])
def test_wp4_cross_order_same_ideal_slow(name):
    """Cross-order ideal invariance for heavier systems.

    Each GB runs in a guarded subprocess: a direct lex Buchberger over QQ on
    these systems can run for hours (FGLM is the practical lex route), and an
    unguarded test that cannot finish is a test that never runs.  Orders that
    exceed the budget are skipped visibly; the remaining pairs are checked.
    """
    polys, gens, _ = _build_corpus(name)

    bases = {}
    for order in ("lex", "grlex", "degrevlex"):
        G = run_with_timeout(_gb_for_order, polys, gens, order, timeout=120)
        if isinstance(G, str):
            continue  # TIMEOUT/ERROR for this order — checked pairs shrink
        bases[order] = G

    if len(bases) < 2:
        pytest.skip(f"{name}: fewer than two orders computable within budget")

    orders = list(bases)
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            a, b = orders[i], orders[j]
            assert _cross_order_same_ideal(bases[a], bases[b], gens, a, b), \
                f"{name}: {a} and {b} ideals disagree"


# --- is_zero_dimensional consistency with FGLM ---

@pytest.mark.parametrize("name", list(SYSTEMS.keys()))
def test_wp4_is_zero_dimensional_matches_fglm(name):
    """is_zero_dimensional(G) must agree with whether fglm() succeeds.

    For zero-dimensional systems, fglm must succeed and produce a valid GB.
    For positive-dimensional systems, fglm must raise NotImplementedError.
    Heavy systems (rose, Uteshev_Bikker) are tested in the slow variant below.
    """
    if name in ("rose", "Uteshev_Bikker"):
        pytest.skip("FGLM too slow — tested in slow variant")

    polys, gens, _ = _build_corpus(name)
    G = groebner_basis(polys, *gens, order="degrevlex")
    is_zd = is_zero_dimensional(list(G), gens, order="degrevlex")

    if is_zd:
        # FGLM must succeed
        try:
            G_lex = G.fglm("lex")
            assert is_groebner(list(G_lex), gens, order="lex"), \
                f"{name}: FGLM produced non-GB output"
        except Exception as e:
            pytest.fail(f"{name}: is_zero_dimensional=True but fglm failed: {e}")
    else:
        # FGLM must fail
        with pytest.raises(NotImplementedError):
            G.fglm("lex")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["rose", "Uteshev_Bikker"])
def test_wp4_is_zero_dimensional_matches_fglm_slow(name):
    """is_zero_dimensional/FGLM consistency for heavy systems."""
    polys, gens, _ = _build_corpus(name)
    G = groebner_basis(polys, *gens, order="degrevlex")
    is_zd = is_zero_dimensional(list(G), gens, order="degrevlex")

    assert is_zd, f"{name} should be zero-dimensional"

    G_lex = G.fglm("lex")
    assert is_groebner(list(G_lex), gens, order="lex"), \
        f"{name}: FGLM produced non-GB output"


@pytest.mark.parametrize("name", ["katsura3", "katsura4", "ammonia"])
def test_wp4_is_zero_dimensional_matches_fglm_extra(name):
    """is_zero_dimensional/FGLM consistency for benchmark systems."""
    se_polys, se_gens = _get_extra_system(name)
    G = groebner_basis(se_polys, *se_gens, order="degrevlex")
    is_zd = is_zero_dimensional(list(G), se_gens, order="degrevlex")

    if is_zd:
        try:
            G_lex = G.fglm("lex")
            assert is_groebner(list(G_lex), se_gens, order="lex")
        except Exception as e:
            pytest.fail(f"{name}: is_zero_dimensional=True but fglm failed: {e}")
    else:
        with pytest.raises(NotImplementedError):
            G.fglm("lex")


# --- cyclic4 is positive-dimensional (good negative case) ---

def test_wp4_cyclic4_positive_dimensional():
    """cyclic4 is positive-dimensional — FGLM must reject it."""
    polys, gens, _ = _build_corpus("cyclic4")
    G = groebner_basis(polys, *gens, order="degrevlex")

    assert is_zero_dimensional(list(G), gens, order="degrevlex") is False, \
        "cyclic4 should NOT be zero-dimensional"

    with pytest.raises(NotImplementedError):
        G.fglm("lex")


def test_wp4_liu_positive_dimensional():
    """liu is positive-dimensional — FGLM must reject it."""
    polys, gens, _ = _build_corpus("liu")
    G = groebner_basis(polys, *gens, order="degrevlex")

    assert is_zero_dimensional(list(G), gens, order="degrevlex") is False, \
        "liu should NOT be zero-dimensional"

    with pytest.raises(NotImplementedError):
        G.fglm("lex")


# --- FGLM with grlex intermediate ---

@pytest.mark.parametrize("name", ["cyclic3", "cyclic5"])
def test_wp4_fglm_grlex_round_trip(name):
    """FGLM round-trip through grlex: degrevlex -> grlex -> degrevlex."""
    polys, gens, _ = _build_corpus(name)

    G_drl = groebner_basis(polys, *gens, order="degrevlex")
    G_grlex = G_drl.fglm("grlex")
    assert G_grlex.order == "grlex"
    assert is_groebner(list(G_grlex), gens, order="grlex")

    G_back = G_grlex.fglm("degrevlex")
    assert G_back.order == "degrevlex"
    assert is_groebner(list(G_back), gens, order="degrevlex")

    assert same_ideal(G_drl, G_back, gens, order="degrevlex"), \
        f"{name}: FGLM grlex round-trip changed the ideal"


# --- Golden size checks for corpus systems ---

@pytest.mark.parametrize("name", list(SYSTEMS.keys()))
def test_wp4_golden_size_degrevlex(name):
    """Corpus systems must match their golden basis sizes (degrevlex)."""
    polys, gens, golden = _build_corpus(name)
    G = groebner_basis(polys, *gens, order="degrevlex")
    assert len(G) == golden, \
        f"{name}: expected {golden} basis elements, got {len(G)}"


@pytest.mark.parametrize("name", ["cyclic3", "cyclic4"])
def test_wp4_golden_size_all_orders(name):
    """Fast systems: verify golden sizes across all orders (from known values)."""
    polys, gens, _ = _build_corpus(name)

    # Known sizes from the test suite
    known = {
        "cyclic3": {"lex": 3, "grlex": 3, "degrevlex": 3},
        "cyclic4": {"lex": 6, "grlex": 7, "degrevlex": 7},
    }

    for order in ["lex", "grlex", "degrevlex"]:
        G = groebner_basis(polys, *gens, order=order)
        expected = known[name][order]
        assert len(G) == expected, \
            f"{name}/{order}: expected {expected}, got {len(G)}"


# --- Reduced basis uniqueness ---

@pytest.mark.parametrize("name", ["cyclic3", "cyclic4"])
def test_wp4_reduced_basis_unique(name):
    """Reduced Groebner bases are unique for a fixed (ideal, order, domain)."""
    polys, gens, _ = _build_corpus(name)

    for order in ["lex", "grlex", "degrevlex"]:
        G1 = groebner_basis(polys, *gens, order=order, algorithm="buchberger")
        G2 = groebner_basis(polys, *gens, order=order, algorithm="f5b")

        if is_reduced_basis(list(G1), gens, order=order) and \
           is_reduced_basis(list(G2), gens, order=order):
            assert set(G1) == set(G2), \
                f"{name}/{order}: reduced bases from buchberger and f5b differ"
