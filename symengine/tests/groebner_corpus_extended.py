"""Extended vendored Groebner benchmark systems.

Mined from msolve, gamba, and other reference resources.
See docs/reports/22a-EXTEND-TESTING_mimov25pro.md for provenance.

golden_size values from verified msolve/gamba output or cross-checked
algorithms (buchberger vs f5b agreement).
"""

# QQ (characteristic 0) systems — fill a major gap in the original corpus.
EXTENDED_SYSTEMS_QQ = {
    "cyclic5_xvars": {
        "gens": ["x1", "x2", "x3", "x4", "x5"],
        "polys": [
            "x1+x2+x3+x4+x5",
            "x1*x2+x1*x5+x2*x3+x3*x4+x4*x5",
            "x1*x2*x3+x1*x2*x5+x1*x4*x5+x2*x3*x4+x3*x4*x5",
            "x1*x2*x3*x4+x1*x2*x3*x5+x1*x2*x4*x5+x1*x3*x4*x5+x2*x3*x4*x5",
            "x1*x2*x3*x4*x5-1",
        ],
        "golden_size": 20,
        "source": "msolve/cyclic5-qq.ms",
        "notes": "cyclic-5 with safe x-var names (no 'e' hazard)",
    },
    "eco6": {
        "gens": ["x0", "x1", "x2", "x3", "x4", "x5"],
        "polys": [
            "x0*x1*x5+x1*x2*x5+x2*x3*x5+x3*x4*x5+x0*x5-1",
            "x0*x2*x5+x1*x3*x5+x2*x4*x5+x1*x5-2",
            "x0*x3*x5+x1*x4*x5+x2*x5-3",
            "x0*x4*x5+x3*x5-4",
            "x4*x5-5",
            "x0+x1+x2+x3+x4+1",
        ],
        "golden_size": 18,
        "source": "msolve/eco6-qq.ms",
        "notes": "Eco-6 benchmark, zero-dimensional",
    },
    "one_unit_ideal": {
        "gens": ["x", "y", "z"],
        "polys": [
            "-62*x**2+97*y**2-73*y*z-56*y+87",
            "-44*x*y+71*x*z-17*y*z+62*x-82*y+80*z",
            "37*x**2-23*x*y+87*x*z+74*y+72*z+6",
            "-47*x**2+40*y**2-81*y*z+91*z**2+11*x-49*z",
        ],
        "golden_size": 1,
        "source": "msolve/one-qq.ms",
        "notes": "inconsistent system, GB = {1} (unit ideal)",
    },
    "quadratic_nonradical": {
        "gens": ["x", "y"],
        "polys": [
            "x**2",
            "y**2",
        ],
        "golden_size": 2,
        "source": "msolve/quadratic-nonradical-qq.ms",
        "notes": "non-radical ideal: x^2,y^2 already reduced GB",
    },
    "multy": {
        "gens": ["x", "y"],
        "polys": [
            "x-y**2",
            "y**3-y",
        ],
        "golden_size": 3,
        "source": "msolve/multy-qq.ms",
        "notes": "4 solutions: (0,0),(1,1),(1,-1), x=y^2",
    },
    "nonradical_radicalshape": {
        "gens": ["x", "y"],
        "polys": [
            "y**4-2*y**2+1",
            "x*y**2+3*y**3-2*x*y-6*y**2+x+3*y",
            "9*y**3+x**2-6*x*y-9*y**2+9*y",
        ],
        "golden_size": 4,
        "source": "msolve/nonradical-radicalshape-qq.ms",
        "notes": "non-radical: y^4-2y^2+1 = (y^2-1)^2",
    },
    "radical_shape": {
        "gens": ["x", "y", "z"],
        "polys": [
            "-2*z**3-3*z**2+x-5*z-7",
            "-z**3+z**2+y-z+1",
            "z**4-z**3-z**2-z-1",
        ],
        "golden_size": 6,
        "source": "msolve/radical-shape-qq.ms",
        "notes": "radical shape, zero-dimensional",
    },
    "univariate_cubic": {
        "gens": ["x"],
        "polys": [
            "x**3-1",
        ],
        "golden_size": 1,
        "source": "msolve/issue-310.ms",
        "notes": "univariate edge case",
    },
    "elimination_5var": {
        "gens": ["t", "w", "x", "y", "z"],
        "polys": [
            "w**4",
            "x**4",
            "w*y**3-x*z**3",
            "t*z-1",
        ],
        "golden_size": 7,
        "source": "msolve/elim-qq.ms",
        "notes": "elimination ideal, tests lex order (not zero-dimensional)",
    },
    "multilinear_trivariate": {
        "gens": ["z", "y", "x"],
        "polys": [
            "-24+6*z+8*y-2*y*z+12*x-3*x*z-4*x*y+x*y*z",
            "-210+42*z+30*y-6*y*z+35*x-7*x*z-5*x*y+x*y*z",
            "-132+44*z+12*y-4*y*z+33*x-11*x*z-3*x*y+x*y*z",
        ],
        "golden_size": 6,
        "source": "msolve/issue-230.ms",
        "notes": "multilinear system, each term is xyz-degree 3",
    },
    "cyclic4_qq": {
        "gens": ["a", "b", "c", "d"],
        "polys": [
            "a+b+c+d",
            "a*b+b*c+c*d+d*a",
            "a*b*c+b*c*d+c*d*a+d*a*b",
            "a*b*c*d-1",
        ],
        "golden_size": 7,
        "source": "gamba/cyclic4-0.txt",
        "notes": "cyclic-4 QQ (same as corpus but QQ verification)",
    },
    "mogvw_basic": {
        "gens": ["a", "b", "c"],
        "polys": [
            "a*b*c - 1",
            "a*b - c",
            "b*c - b",
        ],
        "golden_size": 2,
        "source": "groebner/moGVWTest.cpp",
        "notes": "basic moGVW C++ test",
    },
    "F5_basic": {
        "gens": ["a", "b", "c", "t"],
        "polys": [
            "a*b*c",
            "a*b - c",
            "b*c - b",
        ],
        "golden_size": None,
        "source": "groebner/F5Test.cpp",
        "notes": "basic F5 C++ test",
    },
    "F5_hcyclic3": {
        "gens": ["a", "b", "c", "t"],
        "polys": [
            "a + b + c",
            "a*b + a*c + b*c",
            "a*b*c - t**3",
        ],
        "golden_size": 3,
        "source": "groebner/F5Test.cpp",
        "notes": "homogenized cyclic-3 F5 C++ test",
    },
    "so_snippet_02_full": {
        "gens": ["x", "y"],
        "polys": [
            "(y + Rational(3, 50))*x*(2*x**2 - Rational(6, 25)*y + 2) - 121*x*y",
            "(x**2 - Rational(3, 25)*y + 1)**2 - 4*(y + Rational(3, 50))**2*x**2 - 121*x**2",
        ],
        "golden_size": 5,
        "source": "resources/stackoverflow-snippet-02.md",
        "notes": "Full system from StackOverflow snippet 02",
    },
    "so_snippet_02_factored": {
        "gens": ["x", "y"],
        "polys": [
            "(y + Rational(3, 50))*(2*x**2 - Rational(6, 25)*y + 2) - 121*y",
            "(x**2 - Rational(3, 25)*y + 1)**2 - 4*(y + Rational(3, 50))**2*x**2 - 121*x**2",
        ],
        "golden_size": 3,
        "source": "resources/stackoverflow-snippet-02.md",
        "notes": "Factored system from StackOverflow snippet 02",
    },
}

# GF(p) systems — various primes for coverage.
EXTENDED_SYSTEMS_GFP = {
    "katsura6_gf1073741827": {
        "gens": ["x1", "x2", "x3", "x4", "x5", "x6"],
        "polys": [
            "x1+2*x2+2*x3+2*x4+2*x5+2*x6-1",
            "x1**2+2*x2**2+2*x3**2+2*x4**2+2*x5**2+2*x6**2-x1",
            "2*x1*x2+2*x2*x3+2*x3*x4+2*x4*x5+2*x5*x6-x2",
            "x2**2+2*x1*x3+2*x2*x4+2*x3*x5+2*x4*x6-x3",
            "2*x2*x3+2*x1*x4+2*x2*x5+2*x3*x6-x4",
            "x3**2+2*x2*x4+2*x1*x5+2*x2*x6-x5",
        ],
        "golden_size": 22,
        "modulus": 1073741827,
        "source": "msolve/kat6-31.ms",
        "notes": "Katsura-6 over 30-bit prime",
    },
    "eco6_gf65521": {
        "gens": ["x0", "x1", "x2", "x3", "x4", "x5"],
        "polys": [
            "x0*x1*x5+x1*x2*x5+x2*x3*x5+x3*x4*x5+x0*x5-1",
            "x0*x2*x5+x1*x3*x5+x2*x4*x5+x1*x5-2",
            "x0*x3*x5+x1*x4*x5+x2*x5-3",
            "x0*x4*x5+x3*x5-4",
            "x4*x5-5",
            "x0+x1+x2+x3+x4+1",
        ],
        "golden_size": 18,
        "modulus": 65521,
        "source": "msolve/eco6-16.ms",
        "notes": "Eco-6 over 16-bit prime",
    },
    "coppersmith_gf3": {
        "gens": ["x1", "x2", "x3", "x4"],
        "polys": [
            "-7*x1**3+22*x1**2*x2-56*x1*x2**2+80*x2**3-55*x1**2*x3-44*x2**2*x3-73*x1*x3**2-75*x2*x3**2+23*x3**3-94*x1**2*x4-62*x1*x2*x4+71*x2**2*x4-4*x1*x3*x4-10*x2*x3*x4+75*x3**2*x4-10*x1*x4**2-40*x2*x4**2+6*x3*x4**2+37*x4**3+87*x1**2+97*x1*x2-17*x2**2-83*x1*x3-7*x2*x3-92*x3**2+62*x1*x4+42*x2*x4+74*x3*x4-23*x4**2-82*x1-50*x2+72*x3+87*x4+44",
            "29*x1**3+98*x1**2*x2-8*x1*x2**2-10*x2**3-23*x1**2*x3-29*x1*x2*x3+31*x2**2*x3-49*x1*x3**2+95*x2*x3**2+30*x3**3+10*x1**2*x4+95*x1*x2*x4-51*x2**2*x4-47*x1*x3*x4+x2*x3*x4-27*x3**2*x4-81*x1*x4**2+55*x2*x4**2-59*x3*x4**2-87*x4**3-61*x1**2+11*x1*x2+77*x2**2+40*x1*x3+x2*x3-15*x3**2+91*x1*x4-28*x2*x4-96*x3*x4+47*x4**2+68*x1+16*x2+72*x3-90*x4+43",
            "4884*x1**4+1058*x1**3*x2+1072*x1**2*x2**2-11136*x1*x2**3+6120*x2**4+10557*x1**3*x3+32954*x1**2*x2*x3-50304*x1*x2**2*x3+41100*x2**3*x3-2066*x1**2*x3**2+6525*x1*x2*x3**2+18105*x2**2*x3**2+23221*x1*x3**3-12198*x2*x3**3-13305*x3**4+6009*x1**3*x4-876*x1**2*x2*x4-13532*x1*x2**2*x4-146*x2**3*x4+4227*x1**2*x3*x4-13506*x1*x2*x3*x4+3486*x2**2*x3*x4-2754*x1*x3**2*x4+14395*x2*x3**2*x4-11169*x3**3*x4+5353*x1**2*x4**2+1670*x1*x2*x4**2-13678*x2**2*x4**2+1696*x1*x3*x4**2+13102*x2*x3*x4**2-3150*x3**2*x4**2+5188*x1*x4**3-7256*x2*x4**3-5506*x3*x4**3+2030*x4**4+7388*x1**3+1339*x1**2*x2+11363*x1*x2**2+5752*x2**3+7281*x1**2*x3+47938*x1*x2*x3+4093*x2**2*x3+8352*x1*x3**2-228*x2*x3**2+19031*x3**3-10980*x1**2*x4+10956*x1*x2*x4-19816*x2**2*x4+6410*x1*x3*x4-36652*x2*x3*x4+6594*x3**2*x4-5988*x1*x4**2-4895*x2*x4**2+14545*x3*x4**2-2540*x4**3+1351*x1**2+6134*x1*x2+19638*x2**2+8241*x1*x3+11456*x2*x3-17450*x3**2-19660*x1*x4+9392*x2*x4-6306*x3*x4-5946*x4**2+5520*x1-13474*x2+3868*x3+8656*x4-4752",
            "9432*x1**4+5542*x1**3*x2-20132*x1**2*x2**2+27788*x1*x2**3-10110*x2**4-3368*x1**3*x3+9352*x1**2*x2*x3-14129*x1*x2**2*x3+26*x2**3*x3+120*x1**2*x3**2+7083*x1*x2*x3**2-6618*x2**2*x3**2+6080*x1*x3**3-1399*x2*x3**3-5100*x3**4+6706*x1**3*x4+19916*x1**2*x2*x4-49497*x1*x2**2*x4+24000*x2**3*x4-1064*x1**2*x3*x4+19502*x1*x2*x3*x4-33119*x2**2*x3*x4+9421*x1*x3**2*x4+12806*x2*x3**2*x4+7905*x3**3*x4+94*x1**2*x4**2+6354*x1*x2*x4**2-53715*x2**2*x4**2+13135*x1*x3*x4**2+44*x2*x3*x4**2+7153*x3**2*x4**2+13217*x1*x4**3-25740*x2*x4**3+6559*x3*x4**3+4335*x4**4-2070*x1**3+10101*x1**2*x2+28098*x1*x2**2-14660*x2**3-12057*x1**2*x3+3120*x1*x2*x3-19168*x2**2*x3-13680*x1*x3**2-8659*x2*x3**2+284*x3**3-22662*x1**2*x4+28120*x1*x2*x4+29914*x2**2*x4-15894*x1*x3*x4-10030*x2*x3*x4-14*x3**2*x4-42410*x1*x4**2+1626*x2*x4**2-4028*x3*x4**2-9084*x4**3-1357*x1**2-8106*x1*x2-23092*x2**2-5888*x1*x3-5342*x2*x3-767*x3**2+20917*x1*x4-4238*x2*x4+3949*x3*x4+12749*x4**2-15229*x1-9610*x2+4159*x3-5308*x4+3108",
            "-2712*x1**4-9377*x1**3*x2+5114*x1**2*x2**2-199*x1*x2**3+43*x2**4-8179*x1**3*x3-3987*x1**2*x2*x3+13836*x1*x2**2*x3-5574*x2**3*x3+18830*x1**2*x3**2+20994*x1*x2*x3**2-9296*x2**2*x3**2+8409*x1*x3**3-9231*x2*x3**3-8613*x3**4+3992*x1**3*x4-11770*x1**2*x2*x4+8081*x1*x2**2*x4-1921*x2**3*x4+24882*x1**2*x3*x4+15450*x1*x2*x3*x4-296*x2**2*x3*x4+5393*x1*x3**2*x4+30015*x2*x3**2*x4-9222*x3**3*x4+11130*x1**2*x4**2-2429*x1*x2*x4**2+10906*x2**2*x4**2+24122*x1*x3*x4**2+30824*x2*x3*x4**2-40788*x3**2*x4**2+4109*x1*x4**3-1561*x2*x4**3-33156*x3*x4**3+4983*x4**4-649*x1**3-1077*x1**2*x2-3918*x1*x2**2+216*x2**3-827*x1**2*x3-33770*x1*x2*x3+9667*x2**2*x3+29736*x1*x3**2-5742*x2*x3**2-6066*x3**3+2284*x1**2*x4-3918*x1*x2*x4-120*x2**2*x4+37288*x1*x3*x4-29118*x2*x3*x4+27496*x3**2*x4+4981*x1*x4**2+3592*x2*x4**2+59180*x3*x4**2-10808*x4**3+4406*x1**2+13772*x1*x2-7367*x2**2+8694*x1*x3+4772*x2*x3-1500*x3**2+949*x1*x4+15841*x2*x4-36838*x3*x4-19651*x4**2+6078*x1-4497*x2+6930*x3+11772*x4-12744",
        ],
        "golden_size": None,  # to be determined by cross-check
        "modulus": 3,
        "source": "gamba/cp_d_3_n_4_p_2-2.txt",
        "notes": "Coppersmith problem mod 3, small characteristic edge case",
    },
    "sparse_random_gf32003": {
        "gens": ["x", "y", "z", "t", "u", "v"],
        "polys": [
            "-82*x**2*t-97*x*z+26*y*t*u+7*y*t*v-27*t*v+86*u*v",
            "31*x*y*v-21*y**2*u+91*y*z*u+88*y*u**2-66*z**2+76*t*u*v",
            "12*x**2*t+72*x**2+43*x*t*v+71*y**2*u+8*y*t**2-96*v**3",
            "48*x*t**2+41*y*t+59*y*u**2-58*z*t-90*z*u*v+46*t*u**2",
            "66*x**2*z-62*x*y*z+32*x*t*u+48*x*t*u+82*x*t+52*y**2*u",
            "60*x*z*t-92*z**2*t+97*z**2*v-42*z+22*t**3+52*u",
        ],
        "golden_size": None,  # to be determined by cross-check
        "modulus": 32003,
        "source": "gamba/alea6-15.txt",
        "notes": "random sparse 6-var system over GF(32003)",
    },
    "branch_2_3_gf32003": {
        "gens": ["t", "x", "y"],
        "polys": [
            "x - t**2",
            "y - t**3",
        ],
        "golden_size": 3,
        "modulus": 32003,
        "source": "gamba/branch_2_3-15.txt",
        "notes": "branch 2 3 example",
    },
    "noon3_gf32003": {
        "gens": ["x1", "x2", "x3"],
        "polys": [
            "10*x1**2*x3 + 10*x2**2*x3 - 11*x3 + 10",
            "10*x1*x2**2 + 10*x1*x3**2 - 11*x1 + 10",
            "10*x1**2*x2 + 10*x2*x3**2 - 11*x2 + 10",
        ],
        "golden_size": 11,
        "modulus": 32003,
        "source": "gamba/noon3-15.txt",
        "notes": "noon3 example",
    },
    "minrank_d2_p2_q3_r2_gf32003": {
        "gens": ["x1", "x2"],
        "polys": [
            "350*x1**4 - 2315*x1**3*x2 - 41*x1**2*x2**2 - 10095*x1*x2**3 + 8137*x2**4 + 3874*x1**3 + 2580*x1**2*x2 - 2688*x1*x2**2 - 11516*x2**3 - 2284*x1**2 + 6754*x1*x2 + 1043*x2**2 - 13174*x1 + 2950*x2 - 658",
            "-674*x1**4 + 1629*x1**3*x2 - 623*x1**2*x2**2 + 4870*x1*x2**3 - 7618*x2**4 - 5293*x1**3 - 3263*x1**2*x2 + 970*x1*x2**2 + 6325*x2**3 - 2163*x1**2 - 6854*x1*x2 - 8393*x2**2 + 3847*x1 + 4747*x2 - 4606",
            "-500*x1**4 - 1134*x1**3*x2 - 5896*x1**2*x2**2 - 4231*x1*x2**3 + 1009*x2**4 + 3634*x1**3 - 177*x1**2*x2 - 5662*x1*x2**2 - 8088*x2**3 + 2233*x1**2 - 2938*x1*x2 - 8638*x2**2 + 5465*x1 - 938*x2 - 7661",
        ],
        "golden_size": 5,
        "modulus": 32003,
        "source": "gamba/minrank_d_2_p_2_q_3_r_2-15.txt",
        "notes": "minrank example",
    },
}

# Systems with reserved-name hazards (variable 'e' conflicts with SymEngine E).
# These use explicit Symbol() construction, never eval/string parsing.
EXTENDED_SYSTEMS_RESERVED_HAZARD = {
    "cyclic5_e_hazard_qq": {
        "gens": ["a", "b", "c", "d", "e"],
        "polys": [
            "a+b+c+d+e",
            "a*b+b*c+c*d+d*e+e*a",
            "a*b*c+b*c*d+c*d*e+d*e*a+e*a*b",
            "a*b*c*d+b*c*d*e+c*d*e*a+d*e*a*b+e*a*b*c",
            "a*b*c*d*e-1",
        ],
        "golden_size": 20,
        "source": "gamba/cyclic5-0.txt",
        "notes": "RESERVED HAZARD: variable 'e' — must use explicit Symbol()",
    },
}
