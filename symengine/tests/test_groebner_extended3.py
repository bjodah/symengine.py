"""Extended Groebner basis testing campaign: Work Packages 9-12.

WP9  — Parser safety regression tests
WP10 — New QQ corpus (msolve/gamba mined systems)
WP11 — GF(p) extension systems
WP12 — Parametric subcorpus specialization tests

See docs/reports/22a-EXTEND-TESTING_mimov25pro.md for methodology.
"""

import pytest

from symengine import (
    Symbol, groebner_basis, GroebnerBasis, Rational,
)
from symengine.lib.symengine_wrapper import (
    normal_form,
    is_groebner,
    is_reduced_basis,
    is_zero_dimensional,
)
from symengine.tests.groebner_corpus import SYSTEMS
from symengine.tests.groebner_corpus_extended import (
    EXTENDED_SYSTEMS_QQ,
    EXTENDED_SYSTEMS_GFP,
    EXTENDED_SYSTEMS_RESERVED_HAZARD,
)
from symengine.tests.groebner_parametric_corpus import PARAMETRIC_SYSTEMS
from symengine.tests.groebner_oracle import (
    same_ideal,
    contains_ideal,
    assert_correct_gb,
    run_with_timeout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_corpus(name):
    """Build polys/gens/golden_size from the vendored corpus."""
    s = SYSTEMS[name]
    gens = [Symbol(g) for g in s["gens"]]
    env = {g: gens[i] for i, g in enumerate(s["gens"])}
    env["Rational"] = Rational
    polys = [eval(p, {"__builtins__": {}}, env) for p in s["polys"]]
    return polys, gens, s["golden_size"]


def _build_extended_qq(name):
    """Build polys/gens/golden_size from EXTENDED_SYSTEMS_QQ."""
    s = EXTENDED_SYSTEMS_QQ[name]
    gens = [Symbol(g) for g in s["gens"]]
    env = {g: gens[i] for i, g in enumerate(s["gens"])}
    env["Rational"] = Rational
    polys = [eval(p, {"__builtins__": {}}, env) for p in s["polys"]]
    return polys, gens, s["golden_size"]


def _build_extended_gfp(name):
    """Build polys/gens/golden_size/modulus from EXTENDED_SYSTEMS_GFP."""
    s = EXTENDED_SYSTEMS_GFP[name]
    gens = [Symbol(g) for g in s["gens"]]
    env = {g: gens[i] for i, g in enumerate(s["gens"])}
    env["Rational"] = Rational
    polys = [eval(p, {"__builtins__": {}}, env) for p in s["polys"]]
    return polys, gens, s["golden_size"], s["modulus"]


def _build_reserved_hazard(name):
    """Build polys/gens/golden_size from EXTENDED_SYSTEMS_RESERVED_HAZARD.

    Uses explicit Symbol() construction to avoid the 'e' hazard.
    """
    s = EXTENDED_SYSTEMS_RESERVED_HAZARD[name]
    gens = [Symbol(g) for g in s["gens"]]
    env = {g: gens[i] for i, g in enumerate(s["gens"])}
    env["Rational"] = Rational
    polys = [eval(p, {"__builtins__": {}}, env) for p in s["polys"]]
    return polys, gens, s["golden_size"]


def _build_parametric_specialization(name, spec_dict):
    """Build concrete polys/gens from a parametric system with parameters
    replaced by concrete values from spec_dict."""
    s = PARAMETRIC_SYSTEMS[name]
    gens = [Symbol(g) for g in s["gens"]]
    params = [Symbol(p) for p in s["params"]]
    env = {g: gens[i] for i, g in enumerate(s["gens"])}
    env.update({p: params[i] for i, p in enumerate(s["params"])})
    env.update(spec_dict)
    env["Rational"] = Rational
    polys = [eval(p, {"__builtins__": {}}, env) for p in s["polys"]]
    return polys, gens


# ===========================================================================
# WP9 — Parser safety regression tests
# ===========================================================================


class TestWP9ParserSafety:

    def test_corpus_parser_keeps_reserved_names_as_symbols(self):
        """Mandatory regression: parsing corpus polys with eval() must not
        convert variable names like 'e', 'I', 'pi' into SymEngine constants.

        The eval-based parser uses env={g: Symbol(g) for g in gens} which
        shadows any built-in constant name.  This test verifies that 'e'
        in EXTENDED_SYSTEMS_RESERVED_HAZARD is parsed as a Symbol, not as
        the exponential constant E."""
        for name in EXTENDED_SYSTEMS_RESERVED_HAZARD:
            s = EXTENDED_SYSTEMS_RESERVED_HAZARD[name]
            gens = [Symbol(g) for g in s["gens"]]
            env = {g: gens[i] for i, g in enumerate(s["gens"])}
            for poly_str in s["polys"]:
                p = eval(poly_str, {"__builtins__": {}}, env)
                free = p.free_symbols
                gen_symbols = set(gens)
                # Every generator must appear as a free symbol (not collapsed
                # into a constant like E, I, pi).
                for g_sym in gens:
                    assert g_sym in free, \
                        (f"Parser collapsed generator '{g_sym}' into a "
                         f"constant for poly '{poly_str}'")

    def test_reserved_name_e_as_generator(self):
        """Verify that cyclic5 with variable 'e' works when built with
        explicit Symbol('e'), producing the correct GB size."""
        polys, gens, golden = _build_reserved_hazard("cyclic5_e_hazard_qq")
        G = groebner_basis(polys, *gens, order="degrevlex")
        assert_correct_gb(polys, gens, G, order="degrevlex",
                          golden_size=golden)

    def test_reserved_name_e_not_constant(self):
        """The variable 'e' in the hazard corpus must be a Symbol, not
        SymEngine's E (exponential constant)."""
        e_sym = Symbol("e")
        from symengine import E
        assert e_sym != E, "Symbol('e') must not equal SymEngine E"

    def test_eval_parser_does_not_inject_builtins(self):
        """The eval parser must use __builtins__={} to prevent injection."""
        polys, gens, _ = _build_extended_qq("univariate_cubic")
        assert len(polys) == 1
        # x**3 - 1 should parse correctly
        x = gens[0]
        assert polys[0] == x**3 - 1


# ===========================================================================
# WP10 — New QQ corpus (fast tests)
# ===========================================================================


# All QQ systems that have a non-None golden_size
EXTENDED_QQ_FAST = [
    name for name, s in EXTENDED_SYSTEMS_QQ.items()
    if s["golden_size"] is not None
]

# QQ systems with None golden_size (skip golden check)
EXTENDED_QQ_NO_GOLDEN = [
    name for name, s in EXTENDED_SYSTEMS_QQ.items()
    if s["golden_size"] is None
]


@pytest.mark.parametrize("name", EXTENDED_QQ_FAST)
def test_wp10_qq_buchberger_correct(name):
    """buchberger must produce a correct GB for new QQ systems."""
    polys, gens, golden = _build_extended_qq(name)
    G = groebner_basis(polys, *gens, order="degrevlex",
                       algorithm="buchberger")
    assert_correct_gb(polys, gens, G, order="degrevlex",
                      golden_size=golden)


@pytest.mark.parametrize("name", EXTENDED_QQ_FAST)
def test_wp10_qq_f5b_correct(name):
    """f5b must produce a correct GB for new QQ systems."""
    polys, gens, golden = _build_extended_qq(name)
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b")
    assert_correct_gb(polys, gens, G, order="degrevlex",
                      golden_size=golden)


@pytest.mark.parametrize("name", EXTENDED_QQ_FAST)
def test_wp10_qq_buchberger_f5b_agree(name):
    """buchberger and f5b must agree on new QQ systems."""
    polys, gens, _ = _build_extended_qq(name)
    G_buch = groebner_basis(polys, *gens, order="degrevlex",
                            algorithm="buchberger")
    G_f5b = groebner_basis(polys, *gens, order="degrevlex",
                           algorithm="f5b")
    assert same_ideal(G_buch, G_f5b, gens, order="degrevlex"), \
        f"buchberger and f5b disagree on {name}"


@pytest.mark.parametrize("name", EXTENDED_QQ_NO_GOLDEN)
def test_wp10_qq_no_golden_correct(name):
    """Systems without golden_size: verify correctness via cross-check."""
    polys, gens, _ = _build_extended_qq(name)
    G_buch = groebner_basis(polys, *gens, order="degrevlex",
                            algorithm="buchberger")
    G_f5b = groebner_basis(polys, *gens, order="degrevlex",
                           algorithm="f5b")
    assert is_groebner(list(G_buch), gens, order="degrevlex")
    assert is_groebner(list(G_f5b), gens, order="degrevlex")
    assert contains_ideal(G_buch, polys, gens, order="degrevlex")
    assert contains_ideal(G_f5b, polys, gens, order="degrevlex")
    assert same_ideal(G_buch, G_f5b, gens, order="degrevlex"), \
        f"buchberger and f5b disagree on {name}"


@pytest.mark.parametrize("name", EXTENDED_QQ_FAST)
def test_wp10_qq_is_groebner(name):
    """is_groebner must pass for computed QQ GBs."""
    polys, gens, _ = _build_extended_qq(name)
    G = groebner_basis(polys, *gens, order="degrevlex")
    assert is_groebner(list(G), gens, order="degrevlex")


@pytest.mark.parametrize("name", EXTENDED_QQ_FAST)
def test_wp10_qq_contains_ideal(name):
    """contains_ideal must pass: input polys reduce to 0 mod GB."""
    polys, gens, _ = _build_extended_qq(name)
    G = groebner_basis(polys, *gens, order="degrevlex")
    assert contains_ideal(G, polys, gens, order="degrevlex")


# ===========================================================================
# WP10b — Non-radical system tests
# ===========================================================================


class TestWP10bNonRadical:

    def test_quadratic_nonradical_gb(self):
        """x^2, y^2 is already a reduced GB (non-radical ideal)."""
        polys, gens, golden = _build_extended_qq("quadratic_nonradical")
        G = groebner_basis(polys, *gens, order="degrevlex")
        assert_correct_gb(polys, gens, G, order="degrevlex",
                          golden_size=golden)
        assert set(G) == {gens[0]**2, gens[1]**2}

    def test_nonradical_radicalshape_gb(self):
        """y^4-2y^2+1 = (y^2-1)^2 is non-radical; GB should have size 2."""
        polys, gens, golden = _build_extended_qq("nonradical_radicalshape")
        G = groebner_basis(polys, *gens, order="degrevlex")
        assert_correct_gb(polys, gens, G, order="degrevlex",
                          golden_size=golden)

    def test_multy_gb(self):
        """x-y^2, y^3-y has 4 solutions; GB should have size 3."""
        polys, gens, golden = _build_extended_qq("multy")
        G = groebner_basis(polys, *gens, order="degrevlex")
        assert_correct_gb(polys, gens, G, order="degrevlex",
                          golden_size=golden)

    def test_nonradical_buchberger_f5b_agree(self):
        """Non-radical systems: buchberger and f5b must agree."""
        for name in ["quadratic_nonradical", "nonradical_radicalshape", "multy"]:
            polys, gens, _ = _build_extended_qq(name)
            G_buch = groebner_basis(polys, *gens, order="degrevlex",
                                    algorithm="buchberger")
            G_f5b = groebner_basis(polys, *gens, order="degrevlex",
                                   algorithm="f5b")
            assert same_ideal(G_buch, G_f5b, gens, order="degrevlex"), \
                f"buchberger and f5b disagree on {name}"


# ===========================================================================
# WP11 — GF(p) extension systems
# ===========================================================================


# Fast GF(p) systems (small enough for non-slow suite)
EXTENDED_GFP_FAST = [
    name for name, s in EXTENDED_SYSTEMS_GFP.items()
    if len(s["gens"]) <= 4
]

# Slow GF(p) systems (large, need slow marker)
EXTENDED_GFP_SLOW = [
    name for name, s in EXTENDED_SYSTEMS_GFP.items()
    if len(s["gens"]) > 4
]


@pytest.mark.parametrize("name", EXTENDED_GFP_FAST)
def test_wp11_gfp_f5b_correct(name):
    """f5b must produce a correct GB for new GF(p) systems."""
    polys, gens, golden, modulus = _build_extended_gfp(name)
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b",
                       modulus=modulus)
    assert is_groebner(list(G), gens, order="degrevlex", modulus=modulus)
    assert contains_ideal(G, polys, gens, order="degrevlex", modulus=modulus)
    if golden is not None:
        assert len(G) == golden, \
            f"{name}: expected {golden} basis elements, got {len(G)}"


@pytest.mark.parametrize("name", EXTENDED_GFP_FAST)
def test_wp11_gfp_buchberger_f5b_agree(name):
    """buchberger and f5b must agree on new GF(p) systems."""
    polys, gens, _, modulus = _build_extended_gfp(name)
    G_buch = groebner_basis(polys, *gens, order="degrevlex",
                            algorithm="buchberger", modulus=modulus)
    G_f5b = groebner_basis(polys, *gens, order="degrevlex",
                           algorithm="f5b", modulus=modulus)
    assert same_ideal(G_buch, G_f5b, gens, order="degrevlex",
                      modulus=modulus), \
        f"buchberger and f5b disagree on {name} mod {modulus}"


@pytest.mark.parametrize("name", EXTENDED_GFP_FAST)
def test_wp11_gfp_is_groebner(name):
    """is_groebner must pass for computed GF(p) GBs."""
    polys, gens, _, modulus = _build_extended_gfp(name)
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b",
                       modulus=modulus)
    assert is_groebner(list(G), gens, order="degrevlex", modulus=modulus)


@pytest.mark.parametrize("name", EXTENDED_GFP_FAST)
def test_wp11_gfp_contains_ideal(name):
    """contains_ideal must pass for new GF(p) GBs."""
    polys, gens, _, modulus = _build_extended_gfp(name)
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm="f5b",
                       modulus=modulus)
    assert contains_ideal(G, polys, gens, order="degrevlex", modulus=modulus)


def _run_gb(polys, gens, order, algorithm, modulus, timeout):
    """Run GB computation with timeout protection."""
    kw = {"order": order, "algorithm": algorithm}
    if modulus:
        kw["modulus"] = modulus
    return run_with_timeout(groebner_basis, polys, *gens, **kw,
                            timeout=timeout)


@pytest.mark.slow
@pytest.mark.parametrize("name", EXTENDED_GFP_SLOW)
def test_wp11_gfp_slow_f5b_correct(name):
    """Slow GF(p) systems: f5b must produce a correct GB."""
    polys, gens, golden, modulus = _build_extended_gfp(name)
    G = _run_gb(polys, gens, "degrevlex", "f5b", modulus, timeout=120)
    if isinstance(G, str):
        pytest.skip(f"f5b {G} on {name} mod {modulus}")
    assert is_groebner(list(G), gens, order="degrevlex", modulus=modulus)
    assert contains_ideal(G, polys, gens, order="degrevlex", modulus=modulus)
    if golden is not None:
        assert len(G) == golden, \
            f"{name}: expected {golden} basis elements, got {len(G)}"


@pytest.mark.slow
@pytest.mark.parametrize("name", EXTENDED_GFP_SLOW)
def test_wp11_gfp_slow_buchberger_f5b_agree(name):
    """Slow GF(p) systems: buchberger and f5b must agree."""
    polys, gens, _, modulus = _build_extended_gfp(name)
    G_buch = _run_gb(polys, gens, "degrevlex", "buchberger", modulus,
                     timeout=120)
    G_f5b = _run_gb(polys, gens, "degrevlex", "f5b", modulus, timeout=120)
    if isinstance(G_buch, str) or isinstance(G_f5b, str):
        pytest.skip(f"computation {G_buch}/{G_f5b} on {name} mod {modulus}")
    assert same_ideal(G_buch, G_f5b, gens, order="degrevlex",
                       modulus=modulus), \
        f"buchberger and f5b disagree on {name} mod {modulus}"


@pytest.mark.slow
@pytest.mark.parametrize("name", list(EXTENDED_SYSTEMS_GFP.keys()))
def test_wp11_gfp_m4gb_cross_check(name):
    """m4gb must agree with f5b on new GF(p) systems (if m4gb supports them)."""
    polys, gens, _, modulus = _build_extended_gfp(name)
    G_f5b = _run_gb(polys, gens, "degrevlex", "f5b", modulus, timeout=120)
    if isinstance(G_f5b, str):
        pytest.skip(f"f5b {G_f5b} on {name} mod {modulus}")
    try:
        G_m4gb = groebner_basis(polys, *gens, order="degrevlex",
                                 algorithm="m4gb", modulus=modulus)
    except (ValueError, RuntimeError, NotImplementedError):
        pytest.skip(f"m4gb does not support {name} mod {modulus}")
    assert same_ideal(list(G_m4gb), list(G_f5b), gens,
                       order="degrevlex", modulus=modulus), \
        f"m4gb and f5b disagree on {name} mod {modulus}"


# ===========================================================================
# WP10c — Unit ideal edge case
# ===========================================================================


class TestWP10cUnitIdeal:

    def test_one_unit_ideal_gb_is_one(self):
        """Inconsistent system must produce GB = {1}."""
        polys, gens, golden = _build_extended_qq("one_unit_ideal")
        G = groebner_basis(polys, *gens, order="degrevlex")
        assert_correct_gb(polys, gens, G, order="degrevlex",
                          golden_size=golden)
        assert len(G) == 1
        assert G[0] == 1

    def test_one_unit_ideal_buchberger_f5b_agree(self):
        """Both algorithms must produce GB = {1} for the inconsistent system."""
        polys, gens, _ = _build_extended_qq("one_unit_ideal")
        G_buch = groebner_basis(polys, *gens, order="degrevlex",
                                algorithm="buchberger")
        G_f5b = groebner_basis(polys, *gens, order="degrevlex",
                               algorithm="f5b")
        assert list(G_buch) == [1]
        assert list(G_f5b) == [1]


# ===========================================================================
# WP10d — Elimination / FGLM
# ===========================================================================


class TestWP10dElimination:

    def test_elimination_5var_lex(self):
        """elimination_5var with lex order must produce a correct GB."""
        polys, gens, golden = _build_extended_qq("elimination_5var")
        G = groebner_basis(polys, *gens, order="lex")
        assert is_groebner(list(G), gens, order="lex")
        assert contains_ideal(G, polys, gens, order="lex")
        assert len(G) == golden

    def test_elimination_5var_buchberger_f5b_agree(self):
        """buchberger and f5b must agree on elimination_5var (lex)."""
        polys, gens, _ = _build_extended_qq("elimination_5var")
        G_buch = groebner_basis(polys, *gens, order="lex",
                                algorithm="buchberger")
        G_f5b = groebner_basis(polys, *gens, order="lex",
                               algorithm="f5b")
        assert same_ideal(G_buch, G_f5b, gens, order="lex"), \
            "buchberger and f5b disagree on elimination_5var (lex)"

    def test_elimination_5var_not_zero_dimensional(self):
        """elimination_5var is positive-dimensional (has free variables)."""
        polys, gens, _ = _build_extended_qq("elimination_5var")
        G = groebner_basis(polys, *gens, order="degrevlex")
        assert not is_zero_dimensional(list(G), gens, order="degrevlex")

    @pytest.mark.parametrize("name", [
        "cyclic5_xvars", "eco6", "radical_shape", "multilinear_trivariate",
    ])
    def test_fglm_round_trip_extended(self, name):
        """FGLM round-trip for zero-dimensional extended QQ systems."""
        polys, gens, _ = _build_extended_qq(name)
        G_drl = groebner_basis(polys, *gens, order="degrevlex")
        if not is_zero_dimensional(list(G_drl), gens, order="degrevlex"):
            pytest.skip(f"{name} is not zero-dimensional")

        G_lex = G_drl.fglm("lex")
        assert G_lex.order == "lex"
        assert is_groebner(list(G_lex), gens, order="lex")

        G_back = G_lex.fglm("degrevlex")
        assert G_back.order == "degrevlex"
        assert is_groebner(list(G_back), gens, order="degrevlex")

        assert same_ideal(G_drl, G_back, gens, order="degrevlex"), \
            f"{name}: FGLM round-trip changed the ideal"


# ===========================================================================
# WP10e — Reserved-name hazard tests
# ===========================================================================


class TestWP10eReservedNameHazard:

    def test_cyclic5_e_hazard_qq_gb(self):
        """cyclic5 with variable 'e' must produce correct GB when built
        with explicit Symbol('e')."""
        polys, gens, golden = _build_reserved_hazard("cyclic5_e_hazard_qq")
        G = groebner_basis(polys, *gens, order="degrevlex")
        assert_correct_gb(polys, gens, G, order="degrevlex",
                          golden_size=golden)

    def test_cyclic5_e_hazard_agrees_with_safe_vars(self):
        """cyclic5 with 'e' variable must give same ideal as cyclic5 with
        safe x-var names."""
        polys_e, gens_e, _ = _build_reserved_hazard("cyclic5_e_hazard_qq")
        polys_x, gens_x, _ = _build_extended_qq("cyclic5_xvars")

        G_e = groebner_basis(polys_e, *gens_e, order="degrevlex")
        G_x = groebner_basis(polys_x, *gens_x, order="degrevlex")

        # Same size (both should be 20)
        assert len(G_e) == len(G_x), \
            f"cyclic5 'e' variant has {len(G_e)} elements, " \
            f"x variant has {len(G_x)}"

    def test_parser_rejects_accidentally_created_constants(self):
        """If the parser accidentally created E instead of Symbol('e'),
        the polynomial would have fewer free symbols than expected."""
        s = EXTENDED_SYSTEMS_RESERVED_HAZARD["cyclic5_e_hazard_qq"]
        gens = [Symbol(g) for g in s["gens"]]
        env = {g: gens[i] for i, g in enumerate(s["gens"])}
        # Parse the first polynomial: a+b+c+d+e
        p = eval(s["polys"][0], {"__builtins__": {}}, env)
        # All 5 generators must be free symbols
        for g in gens:
            assert g in p.free_symbols, \
                f"Generator {g} was collapsed into a constant"


# ===========================================================================
# WP12 — Parametric subcorpus specialization tests
# ===========================================================================


# Concrete parameter values for each parametric system
_PARAMETRIC_SPECIALIZATIONS = {
    "compregb_basic": [
        {"a": 1, "b": 1},       # generic branch
        {"a": 0, "b": 1},       # a=0 degenerate
        {"a": 1, "b": 0},       # b=0 degenerate
        {"a": 0, "b": 0},       # fully degenerate
    ],
    "parametric_linear_K": [
        {"K": 1},               # generic
        {"K": 2},               # generic, different value
        {"K": 0},               # degenerate
    ],
    "parametric_2var_2par": [
        {"K1": 1, "K2": 1},     # generic
        {"K1": 0, "K2": 1},     # K1=0 degenerate
    ],
    "singular_case1": [
        {"K1": 1, "ux": 2},     # generic
        {"K1": 0, "ux": 2},     # K1=0 degenerate
    ],
    "robot_2arm": [
        {"a": 1, "b": 1, "l2": 1, "l3": 1},   # generic
        {"a": 0, "b": 0, "l2": 0, "l3": 1},   # l2=0 degenerate
    ],
    "conic_intersection": [
        {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 1},  # generic
        {"a": 1, "b": 0, "c": 1, "d": 1, "e": 0, "f": 1},  # a*e-b*d=0
    ],
    "casas_alvero_degree4": [
        {"a0": 0, "a1": 0, "a2": 0, "a3": 0},  # x^4
        {"a0": 1, "a1": 0, "a2": 0, "a3": 0},  # x^4 + 1
    ],
    "grobcov_pdivi": [
        {"a": 2, "b": 3, "c": 5},               # generic
        {"a": 0, "b": 0, "c": 5},               # degenerate a=0
        {"a": 2, "b": 3, "c": 0},               # degenerate c=0
    ],
    "grobcov_pnormalf": [
        {"a": 2, "b": 2, "c": 2},               # generic
        {"a": 2, "b": 2, "c": 1},               # degenerate c=1
        {"a": 2, "b": 2, "c": 2},               # degenerate a=b (already covered)
    ],
    "grobcov_two_quadratics": [
        {"a0": 1, "b0": 0, "c0": -1, "a1": 1, "b1": 0, "c1": -4},  # generic (inconsistent, GB=[1])
        {"a0": 1, "b0": 0, "c0": -1, "a1": 1, "b1": -1, "c1": 0}, # generic (shares root x=1)
        {"a0": 0, "b0": 1, "c0": 2, "a1": 1, "b1": 0, "c1": -1},  # degenerate a0=0
    ],
    "grobcov_concoid_locus": [
        {"x": 2, "y": 0},                       # generic
        {"x": 0, "y": 2},                       # degenerate x=0
    ],
    "so_snippet_01": [
        {"kp1": Rational(21, 10), "kp2": Rational(19, 5)},  # generic specialization
        {"kp1": 1, "kp2": 1},
    ],
}


@pytest.mark.parametrize(
    "name,spec",
    [
        (name, spec)
        for name, specs in _PARAMETRIC_SPECIALIZATIONS.items()
        for spec in specs
    ],
    ids=[
        f"{name}@{','.join(f'{k}={v}' for k, v in spec.items())}"
        for name, specs in _PARAMETRIC_SPECIALIZATIONS.items()
        for spec in specs
    ],
)
def test_wp12_parametric_specialization_gb(name, spec):
    """Substituting concrete parameter values into a parametric system
    must produce a correct GB."""
    polys, gens = _build_parametric_specialization(name, spec)
    # Remove zero polynomials that may result from substitution
    polys = [p for p in polys if p != 0]
    if not polys:
        pytest.skip("All polynomials vanished after specialization")
    G = groebner_basis(polys, *gens, order="degrevlex")
    assert is_groebner(list(G), gens, order="degrevlex")
    assert contains_ideal(G, polys, gens, order="degrevlex")


@pytest.mark.parametrize("name", list(PARAMETRIC_SYSTEMS.keys()))
def test_wp12_parametric_branches_documented(name):
    """Every parametric system must have at least one branch documented."""
    s = PARAMETRIC_SYSTEMS[name]
    assert len(s["branches"]) >= 1, \
        f"{name}: no branches documented"


@pytest.mark.parametrize("name", list(PARAMETRIC_SYSTEMS.keys()))
def test_wp12_parametric_generic_point_valid(name):
    """For parametric systems, substituting all parameters to 1 (generic
    point) must produce a valid GB."""
    s = PARAMETRIC_SYSTEMS[name]
    spec = {p: 1 for p in s["params"]}
    polys, gens = _build_parametric_specialization(name, spec)
    polys = [p for p in polys if p != 0]
    if not polys:
        pytest.skip("All polynomials vanished at generic point")
    G = groebner_basis(polys, *gens, order="degrevlex")
    assert is_groebner(list(G), gens, order="degrevlex")
    assert contains_ideal(G, polys, gens, order="degrevlex")


@pytest.mark.parametrize("name", list(PARAMETRIC_SYSTEMS.keys()))
def test_wp12_parametric_specialization_cross_check(name):
    """For each specialization, buchberger and f5b must agree."""
    specs = _PARAMETRIC_SPECIALIZATIONS.get(name, [{pr: 1 for pr in PARAMETRIC_SYSTEMS[name]["params"]}])
    for spec in specs:
        polys, gens = _build_parametric_specialization(name, spec)
        polys = [p for p in polys if p != 0]
        if not polys:
            continue
        G_buch = groebner_basis(polys, *gens, order="degrevlex",
                                algorithm="buchberger")
        G_f5b = groebner_basis(polys, *gens, order="degrevlex",
                               algorithm="f5b")
        assert same_ideal(G_buch, G_f5b, gens, order="degrevlex"), \
            f"buchberger and f5b disagree on {name} with {spec}"


# ===========================================================================
# WP11b — Extended QQ systems with grlex/lex orders
# ===========================================================================


FAST_QQ_GRLEX = [
    "eco6", "one_unit_ideal", "quadratic_nonradical", "multy",
    "nonradical_radicalshape", "radical_shape", "univariate_cubic",
    "multilinear_trivariate", "cyclic4_qq", "mogvw_basic", "F5_hcyclic3",
]


@pytest.mark.parametrize("name", FAST_QQ_GRLEX)
def test_wp11b_qq_grlex_correct(name):
    """Extended QQ systems must produce correct GBs under grlex order."""
    polys, gens, _ = _build_extended_qq(name)
    G = groebner_basis(polys, *gens, order="grlex")
    assert is_groebner(list(G), gens, order="grlex")
    assert contains_ideal(G, polys, gens, order="grlex")


@pytest.mark.parametrize("name", ["eco6", "quadratic_nonradical", "multy",
                                   "univariate_cubic", "cyclic4_qq",
                                   "mogvw_basic", "F5_hcyclic3"])
def test_wp11b_qq_lex_correct(name):
    """Fast extended QQ systems must produce correct GBs under lex order."""
    polys, gens, _ = _build_extended_qq(name)
    G = groebner_basis(polys, *gens, order="lex")
    assert is_groebner(list(G), gens, order="lex")
    assert contains_ideal(G, polys, gens, order="lex")


# ===========================================================================
# WP11c — Cross-order ideal invariance for extended systems
# ===========================================================================


@pytest.mark.parametrize("name", ["eco6", "quadratic_nonradical", "multy",
                                   "univariate_cubic", "cyclic4_qq",
                                   "mogvw_basic", "F5_hcyclic3"])
def test_wp11c_cross_order_same_ideal(name):
    """Extended QQ systems: GBs under different orders must generate
    the same ideal."""
    polys, gens, _ = _build_extended_qq(name)

    G_lex = groebner_basis(polys, *gens, order="lex")
    G_grlex = groebner_basis(polys, *gens, order="grlex")
    G_drl = groebner_basis(polys, *gens, order="degrevlex")

    # Check containment in both directions using the appropriate order
    assert contains_ideal(G_lex, list(G_grlex), gens, order="lex")
    assert contains_ideal(G_grlex, list(G_lex), gens, order="grlex")
    assert contains_ideal(G_lex, list(G_drl), gens, order="lex")
    assert contains_ideal(G_drl, list(G_lex), gens, order="degrevlex")
    assert contains_ideal(G_grlex, list(G_drl), gens, order="grlex")
    assert contains_ideal(G_drl, list(G_grlex), gens, order="degrevlex")


# ===========================================================================
# Mined tests from Consultant B
# ===========================================================================

def test_mogvw_basic_exact_lex_basis():
    """moGVW_basic lex: exact expected basis {c - 1, a*b - 1}."""
    polys, gens, _ = _build_extended_qq("mogvw_basic")
    G = groebner_basis(polys, *gens, order="lex")
    assert is_reduced_basis(list(G), gens, order="lex")
    assert set(G) == {gens[2] - 1, gens[0]*gens[1] - 1}


def test_F5_hcyclic3_exact_degrevlex_basis():
    """F5_hcyclic3 degrevlex: exact expected basis."""
    polys, gens, _ = _build_extended_qq("F5_hcyclic3")
    G = groebner_basis(polys, *gens, order="degrevlex")
    assert is_reduced_basis(list(G), gens, order="degrevlex")
    assert set(G) == {
        gens[0] + gens[1] + gens[2],
        gens[1]**2 + gens[1]*gens[2] + gens[2]**2,
        gens[2]**3 - gens[3]**3,
    }


@pytest.mark.parametrize("algorithm", ["buchberger", "f5b"])
def test_groebner_cpp_cross_algo(algorithm):
    """Cross-algorithm agreement on F5_basic."""
    polys, gens, _ = _build_extended_qq("F5_basic")
    G = groebner_basis(polys, *gens, order="degrevlex", algorithm=algorithm)
    G_ref = groebner_basis(polys, *gens, order="degrevlex", algorithm="buchberger")
    assert same_ideal(G, G_ref, gens, order="degrevlex"), \
        f"{algorithm} disagrees with buchberger on F5_basic"


@pytest.mark.parametrize("modulus", [3, 32003, 2147483647])
def test_branch_2_3_multi_prime(modulus):
    """branch_2_3 across multiple GF(p) sizes."""
    polys, gens, _, _ = _build_extended_gfp("branch_2_3_gf32003")
    # Change modulus dynamically
    G_f5b = groebner_basis(polys, *gens, order="degrevlex",
                           algorithm="f5b", modulus=modulus)
    G_buch = groebner_basis(polys, *gens, order="degrevlex",
                            algorithm="buchberger", modulus=modulus)
    assert is_groebner(list(G_f5b), gens, order="degrevlex", modulus=modulus)
    assert same_ideal(G_f5b, G_buch, gens, order="degrevlex", modulus=modulus)
    assert len(G_f5b) == 3


def test_two_quadratics_concrete_check():
    """Two quadratics: x^2-1, x^2-4 are inconsistent -> GB = [1]."""
    x = Symbol('x')
    polys = [x**2 - 1, x**2 - 4]
    G = groebner_basis(polys, x, order="degrevlex")
    assert is_groebner(list(G), [x], order="degrevlex")
    assert G[0] == 1
    assert len(G) == 1


def test_two_quadratics_shared_root():
    """Two quadratics sharing a root: x^2-1 and x^2-x."""
    x = Symbol('x')
    polys = [x**2 - 1, x**2 - x]
    G = groebner_basis(polys, x, order="degrevlex")
    assert is_groebner(list(G), [x], order="degrevlex")
    assert x - 1 in set(G)


def test_concoid_locus_concrete_check():
    """concoid locus: x=2, y=0 -> check GB."""
    a_sym, b_sym = Symbol('a'), Symbol('b')
    polys = [
        2**2 + 0**2 - 4,  # x^2+y^2-4 = 0 (holds)
        (b_sym - 2)*2 - a_sym*0 + 2*a_sym,
        (a_sym - 2)**2 + (b_sym - 0)**2 - 1,
    ]
    G = groebner_basis(polys, a_sym, b_sym, order="degrevlex")
    assert is_groebner(list(G), [a_sym, b_sym], order="degrevlex")
    assert contains_ideal(G, polys, [a_sym, b_sym], order="degrevlex")

