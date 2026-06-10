"""Vendored parametric Groebner basis test systems.

Mined from Singular's compregb.lib and grobcov.lib examples.
Each system records:
  - gens: generator (variable) names
  - params: parameter names
  - polys: polynomial strings (using gens + params)
  - branches: list of (condition_description, specialization_dict) pairs
    describing the expected comprehensive GB structure.

These are NOT used for automated GB computation (SymEngine does not yet
support comprehensive GB).  They are vendored for:
  (a) parametric GB generic-point tests (where the generic point is valid);
  (b) future comprehensive GB work;
  (c) documentation of branch structure.

Source: Singular 4.4.1 compregb.lib and grobcov.lib examples.
"""

PARAMETRIC_SYSTEMS = {
    "compregb_basic": {
        "gens": ["x", "y", "t"],
        "params": ["a", "b"],
        "polys": [
            "x**3 - a",
            "y**4 - b",
            "x + y - t",
        ],
        "source": "singular/compregb.lib",
        "notes": "Canonical CGS example from Singular. "
                 "Branches on a=0, b=0 (leading coefficient vanishing).",
        "branches": [
            {
                "condition": "a != 0 and b != 0",
                "description": "generic branch: full elimination",
            },
            {
                "condition": "a == 0",
                "description": "degenerate: x^3=0 collapses",
            },
            {
                "condition": "b == 0",
                "description": "degenerate: y^4=0 collapses",
            },
            {
                "condition": "a == 0 and b == 0",
                "description": "fully degenerate",
            },
        ],
    },
    "parametric_linear_K": {
        "gens": ["x", "y", "z"],
        "params": ["K"],
        "polys": [
            "x + y + z - 3",
            "x - y + z - 5",
            "K*z**2 + z - 2",
        ],
        "source": "symcse/test_mpolysys.py",
        "notes": "Simplest parametric system. "
                 "K=0 makes the third equation linear; K!=0 keeps it quadratic.",
        "branches": [
            {
                "condition": "K != 0",
                "description": "generic: quadratic in z, 2 solutions",
            },
            {
                "condition": "K == 0",
                "description": "degenerate: z=2, unique solution",
            },
        ],
    },
    "parametric_2var_2par": {
        "gens": ["x", "y"],
        "params": ["K1", "K2"],
        "polys": [
            "x**2 + K1*x*y",
            "x*y + 2*y**3 - K2",
        ],
        "source": "symcse/test__groebner_sage.py",
        "notes": "Minimal 2-var 2-param system.",
        "branches": [
            {
                "condition": "K1 != 0",
                "description": "generic branch",
            },
            {
                "condition": "K1 == 0",
                "description": "first equation becomes x^2=0",
            },
        ],
    },
    "singular_case1": {
        "gens": ["z", "y", "x"],
        "params": ["K1", "ux"],
        "polys": [
            "y*z - K1*(ux - x)",
            "y - x",
            "z - x",
        ],
        "source": "symcse/test__singular_groebner.py:TestCase1",
        "notes": "Chemical equilibrium, 3 linear+1 bilinear. "
                 "Substituting y=x, z=x gives x^2 - K1*(ux-x) = 0.",
        "branches": [
            {
                "condition": "K1 != 0",
                "description": "generic: quadratic, 2 solutions",
            },
            {
                "condition": "K1 == 0",
                "description": "degenerate: x^2=0, double root at 0",
            },
        ],
    },
    "robot_2arm": {
        "gens": ["c3", "s3", "c1", "s1"],
        "params": ["a", "b", "l2", "l3"],
        "polys": [
            "a - l3*c3 - l2*c1",
            "b - l3*s3 - l2*s1",
            "c1**2 + s1**2 - 1",
            "c3**2 + s3**2 - 1",
        ],
        "source": "singular/grobcov.lib",
        "notes": "Two-arm robot (Rychlik). Classical kinematics. "
                 "c1,s1 = cos/sin of joint 1; c3,s3 = cos/sin of joint 3.",
        "branches": [
            {
                "condition": "l2 != 0 and l3 != 0",
                "description": "generic: 4 solutions (elbow up/down combinations)",
            },
            {
                "condition": "l2 == 0",
                "description": "degenerate: arm 2 has zero length",
            },
            {
                "condition": "l3 == 0",
                "description": "degenerate: arm 3 has zero length",
            },
        ],
    },
    "conic_intersection": {
        "gens": ["x", "y"],
        "params": ["a", "b", "c", "d", "e", "f"],
        "polys": [
            "a*x**2 + b*x*y + c*y**2",
            "d*x**2 + e*x*y + f*y**2",
        ],
        "source": "singular/grobcov.lib/WLemma",
        "notes": "Intersection of two conics through the origin. "
                 "Non-degeneracy: a*e - b*d != 0.",
        "branches": [
            {
                "condition": "a*e - b*d != 0",
                "description": "generic: two distinct lines through origin",
            },
            {
                "condition": "a*e - b*d == 0",
                "description": "degenerate: conics share a component",
            },
        ],
    },
    "casas_alvero_degree4": {
        "gens": ["x1", "x2", "x3"],
        "params": ["a0", "a1", "a2", "a3"],
        "polys": [
            "x1**4 + 4*a3*x1**3 + 6*a2*x1**2 + 4*a1*x1 + a0",
            "x1**3 + 3*a3*x1**2 + 3*a2*x1 + a1",
            "x2**4 + 4*a3*x2**3 + 6*a2*x2**2 + 4*a1*x2 + a0",
            "x2**2 + 2*a3*x2 + a2",
            "x3**4 + 4*a3*x3**3 + 6*a2*x3**2 + 4*a1*x3 + a0",
            "x3 + a3",
        ],
        "source": "singular/grobcov.lib/Casas-Alvero",
        "notes": "Casas-Alvero conjecture, degree 4. "
                 "x1,x2,x3 track roots with multiplicity.",
        "branches": [
            {
                "condition": "generic (parameters in general position)",
                "description": "isolated solutions corresponding to "
                               "polynomials with a root of multiplicity >= 2",
            },
        ],
    },
    "grobcov_pdivi": {
        "gens": ["x", "y"],
        "params": ["a", "b", "c"],
        "polys": [
            "a*x + b",
            "c*y + a",
        ],
        "source": "singular/grobcov.lib",
        "notes": "pdivi example. Branches on leading coefficients.",
        "branches": [
            {
                "condition": "a != 0 and c != 0",
                "description": "generic branch: both leading coeffs nonzero",
            },
            {
                "condition": "a == 0",
                "description": "degenerate: first eq becomes b=0",
            },
            {
                "condition": "c == 0",
                "description": "degenerate: second eq becomes a=0",
            },
        ],
    },
    "grobcov_pnormalf": {
        "gens": ["x", "y"],
        "params": ["a", "b", "c"],
        "polys": [
            "(b**2 - 1)*x**3*y + (c**2 - 1)*x*y**2 + (c**2*b - b)*x + (a - b*c)*y",
        ],
        "source": "singular/grobcov.lib",
        "notes": "pnormalf example. Reduction denominator vanishing.",
        "branches": [
            {
                "condition": "b**2 - 1 != 0 and c**2 - 1 != 0",
                "description": "generic branch: reduction denominator nonzero",
            },
            {
                "condition": "c == 1",
                "description": "degenerate: null ideal when c-1=0",
            },
            {
                "condition": "a == b",
                "description": "degenerate: hole ideal when a-b=0",
            },
        ],
    },
    "grobcov_two_quadratics": {
        "gens": ["x"],
        "params": ["a0", "b0", "c0", "a1", "b1", "c1"],
        "polys": [
            "a0*x**2 + b0*x + c0",
            "a1*x**2 + b1*x + c1",
        ],
        "source": "singular/grobcov.lib",
        "notes": "Two quadratic polynomials in x.",
        "branches": [
            {
                "condition": "a0 != 0 and a1 != 0",
                "description": "generic branch: both leading coeffs nonzero",
            },
            {
                "condition": "a0 == 0",
                "description": "degenerate: first poly linearizes or becomes constant",
            },
            {
                "condition": "a1 == 0",
                "description": "degenerate: second poly linearizes or becomes constant",
            },
        ],
    },
    "grobcov_concoid_locus": {
        "gens": ["a", "b"],
        "params": ["x", "y"],
        "polys": [
            "x**2 + y**2 - 4",
            "(b - 2)*x - a*y + 2*a",
            "(a - x)**2 + (b - y)**2 - 1",
        ],
        "source": "singular/grobcov.lib",
        "notes": "Concoid locus example.",
        "branches": [
            {
                "condition": "x != 0",
                "description": "generic branch: concoid curve with general parameters",
            },
            {
                "condition": "x == 0",
                "description": "degenerate when x=0",
            },
        ],
    },
    "so_snippet_01": {
        "gens": ["a", "b", "d", "e", "f"],
        "params": ["kp1", "kp2"],
        "polys": [
            "a + b - Rational(13, 5)",
            "2*a + b + d + 2*f - 7",
            "d + e - 2",
            "a*e - kp2*b*d",
            "b**2*f - a**2*kp1**2 * (a + b + d + e + f + Rational(329, 25))",
        ],
        "source": "resources/stackoverflow-snippet-01.md",
        "notes": "Parametric system from StackOverflow snippet 01",
        "branches": [
            {
                "condition": "kp1 != 0 and kp2 != 0",
                "description": "generic branch",
            },
        ],
    },
}
