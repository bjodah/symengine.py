"""Stats fix (fallout of P2-4, from review): GroebnerBasis.fglm() used to
carry the *source* basis's Python stats dict forward, only refreshing
genericity_assumptions -- so target_stats['fglm_reductions'] showed the
source computation's 0 instead of the conversion's own work. fglm() now
builds its stats dict from the fglm_convert result's own stats
(_groebner_stats_to_dict(target_result.stats)), which the C++ side
(fglm_impl in groebner.cpp) already populates with FGLM's own counters
(input_polys, output_polys, max_basis_size, fglm_reductions, ...) plus
genericity_assumptions carried forward from the source and unioned with any
new pivots FGLM's own conversion introduces.
"""

from symengine import Symbol, groebner_basis


def test_fglm_reports_its_own_fglm_reductions_not_source():
    x, y = Symbol('x'), Symbol('y')
    source = groebner_basis([x**2 - y, y**2 - x], x, y, order='degrevlex')
    # A plain groebner_basis run never does FGLM work.
    assert source.stats['fglm_reductions'] == 0

    converted = source.fglm('lex')
    assert converted.stats['fglm_reductions'] > 0


def test_fglm_stats_are_fglm_impls_own_not_a_stale_copy():
    """Converting twice in a row must not accumulate or carry forward
    unrelated source counters (e.g. s_pairs_processed from the original
    groebner_basis run) into the fglm result's stats."""
    x, y = Symbol('x'), Symbol('y')
    source = groebner_basis([x**2 - y, y**2 - x], x, y, order='degrevlex')
    converted = source.fglm('lex')

    # fglm_impl builds a fresh GroebnerStats; work counters that only make
    # sense for a Buchberger/F5B/etc. run (not FGLM's staircase loop) must
    # be zero, not inherited from the source.
    assert converted.stats['s_pairs_processed'] == 0
    assert converted.stats['reductions_to_zero'] == 0
    assert converted.stats['input_polys'] == len(source.polys)
    assert converted.stats['output_polys'] == len(converted.polys)


def test_fglm_genericity_assumptions_survive_parametric_conversion():
    """Mirrors test_fglm_records_new_parametric_pivots in
    test_groebner_extended2.py: genericity assumptions recorded by the
    source computation (or newly introduced by FGLM's own pivots) must
    still show up on the converted basis's stats -- the stats-dict rebuild
    must not accidentally drop them while fixing fglm_reductions."""
    x, y, K1, K2 = map(Symbol, ('x', 'y', 'K1', 'K2'))
    F = [x**2 + K1 * x * y, x * y + 2 * y**3 - K2]
    source = groebner_basis(F, x, y, order='degrevlex')
    converted = source.fglm('lex')

    assert source.stats['genericity_assumptions'] == ()
    assert K2 in converted.stats['genericity_assumptions']
    # And the fix under test: this parametric conversion actually did FGLM
    # work, which must be visible too.
    assert converted.stats['fglm_reductions'] > 0


def test_fglm_stats_independent_of_source_stats_solver_counters():
    """Regression for the exact bug: a source basis produced with
    solver-orchestration-flavored stats present (solver_attempts etc. are
    always 0 for a plain groebner_basis call, but the *dict itself* used to
    be carried forward wholesale) must not leak any of its own irrelevant
    counters into the converted basis -- everything in converted.stats
    (other than genericity_assumptions) must come from fglm_impl's own
    fresh GroebnerStats, keyed off the conversion's own work."""
    x, y = Symbol('x'), Symbol('y')
    source = groebner_basis([x**2 - y, y**2 - x], x, y, order='degrevlex')
    converted = source.fglm('lex')
    assert converted.stats['solver_attempts'] == 0
    assert converted.stats['solver_failed_attempts'] == 0
    assert converted.stats['max_basis_size'] >= len(converted.polys)
    assert converted.stats['fglm_reductions'] > 0
