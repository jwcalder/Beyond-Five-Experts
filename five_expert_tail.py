"""Exact one-leader tail of the five-expert solution (Remark 4.12).

Along x = (s, 0, 0, 0, 0) the Region III coordinates of the five-expert
solution [Calder--Drenska, Theorem 2.1] are a1 = a2 = a3 = a4 = a = kappa*s/3,
and u_5(s,0,0,0,0) - s = F/kappa with F given by (2.16)--(2.17) there.
This script checks, in exact rational arithmetic, that

    F = (32/35) e^{-3a} + O(a e^{-5a})   as a -> infinity,

so that u_5(s,0,0,0,0) - s ~ (32/(35*sqrt(2))) e^{-sqrt(2) s} = h_4(0) e^{-sqrt(2) s}.

Method. Write E = e^X. With q = e^{-2t}, the integrand of the trace
quadrature (2.8),

    e(X) = 3 cosh X  int_X^infty I(t) / (cosh^2 t sinh^5 t) dt,

equals (1+q)^{-2} (1-q)^{-5} times
    e^{-t}/3 - e^{-3t} - e^{-5t} + 8 t e^{-7t} + e^{-9t} + e^{-11t} - e^{-13t}/3,
where I(t) is (2.7). Expanding (1+q)^{-2}(1-q)^{-5} in powers of q and
integrating term by term gives e(X) as a series in E^{-2} whose coefficients
are polynomials of degree at most one in X. Derivatives act on X^j E^m by
d/dX = d/dX|_E + E d/dE. Substituting e, e', e'', e''' and
lambda(a) = log coth(a/2) = 2 sum_j e^{-(2j+1)a}/(2j+1) into (2.16)--(2.17)
with a1 = a2 = a3 = a gives F as a Laurent series in E with coefficients
polynomial in X; keeping N orders of q makes every coefficient of E^m with
m >= 3 - 2N exact. On the diagonal, e and its derivatives enter F with
coefficients of order e^{3a}, while lambda enters with the coefficient
(12/7) sinh^3 a cosh a of order e^{4a}. So the e^{-3a} coefficient of F needs
lambda through e^{-7a} (truncating after e^{-3a} would give -3/70 for the
coefficient of e^{-a} and 95/98 for that of e^{-3a}); we keep N + 2 terms.

Two sanity checks at sample points (floating point, not part of the
argument) compare the series for e with the ODE (2.9) and with the quadrature.

Run: python3 five_expert_tail.py   (requires sympy)
"""

import sympy as sp

X = sp.Symbol("X", positive=True)  # the variable of e, and later a
E = sp.Symbol("E", positive=True)  # stands for e^X
q = sp.Symbol("q")
N = 6  # orders of q kept; coefficients of E^m are exact for m >= 3 - 2N

# Expansion of the trace e(X) from the quadrature (2.8).
weights = sp.series((1 + q) ** -2 * (1 - q) ** -5, q, 0, N + 1).removeO()
bracket = [  # (rate r, coefficient of e^{-rt}, coefficient of t e^{-rt})
    (1, sp.Rational(1, 3), 0), (3, -1, 0), (5, -1, 0), (7, 0, 8),
    (9, 1, 0), (11, 1, 0), (13, -sp.Rational(1, 3), 0),
]
tail = 0  # int_X^infty of the integrand, as a series in E^{-1}
for j in range(N + 1):
    w = weights.coeff(q, j)
    for r, c0, c1 in bracket:
        rr = r + 2 * j
        tail += w * E ** (-rr) * (c0 / sp.Integer(rr) + c1 * (X / sp.Integer(rr) + sp.Rational(1, rr ** 2)))
e = sp.expand(sp.Rational(3, 2) * (E + 1 / E) * tail)


def D(f):
    """d/dX on expressions in X and E = e^X."""
    return sp.expand(sp.diff(f, X) + E * sp.diff(f, E))


def coefficients(f):
    """Map (power of X, power of E) -> exact coefficient."""
    shift = 4 * N + 10
    poly = sp.Poly(sp.expand(f * E ** shift), X, E)
    return {(i, k - shift): c for (i, k), c in poly.terms()}


e1, e2, e3 = D(e), D(D(e)), D(D(D(e)))

print("Expansion of e(X) as X -> infinity:")
for (i, k), c in sorted(coefficients(e).items(), key=lambda t: (-t[0][1], t[0][0])):
    if k >= -8:
        print(f"  X^{i} e^({k}X): {c}")
e_coeffs = coefficients(e)
assert e_coeffs[(0, 0)] == sp.Rational(1, 2)
assert e_coeffs[(0, -2)] == sp.Rational(1, 2)
assert e_coeffs[(0, -4)] == -sp.Rational(2, 5)
assert e_coeffs[(1, -6)] == sp.Rational(12, 7)
assert e_coeffs[(0, -6)] == -sp.Rational(671, 490)

# The Region III formula (2.16)-(2.17) on the diagonal a1 = a2 = a3 = a4 = a.
ch, sh = (E + 1 / E) / 2, (E - 1 / E) / 2
lam = 2 * sum(E ** (-(2 * j + 1)) / sp.Integer(2 * j + 1) for j in range(N + 2))
b0 = e
b1 = (e1 - 3) / 6
b2 = (e2 + 6 * e + 18 * lam * sh) / 42
b3 = (e3 + 20 * e1) / 336 + (6 * lam * ch - 13) / 14
F = sp.expand(b0 * ch ** 3 + 3 * b1 * sh * ch ** 2 + 3 * b2 * sh ** 2 * ch + b3 * sh ** 3)
F_coeffs = coefficients(F)

print("Expansion of F along the diagonal, terms down to e^(-5a):")
for (i, k), c in sorted(F_coeffs.items(), key=lambda t: (-t[0][1], t[0][0])):
    if k >= -5 and c != 0:
        print(f"  a^{i} e^({k}a): {c}")

# Exact checks: every term growing at least like e^{-3a} vanishes except (32/35) e^{-3a}.
for (i, k), c in F_coeffs.items():
    if k > -3 or (k == -3 and i > 0):
        assert c == 0, (i, k, c)
assert F_coeffs[(0, -3)] == sp.Rational(32, 35)
h4_0 = sp.Rational(32, 35) / sp.sqrt(2)  # h_4(0) = 8!!/(4 sqrt(2) 7!!)
assert sp.simplify(h4_0 - sp.factorial2(8) / (4 * sp.sqrt(2) * sp.factorial2(7))) == 0
print("Exact: F = (32/35) e^{-3a} + O(a e^{-5a}), so u_5(s,0,0,0,0) - s ~ h_4(0) e^{-sqrt(2) s}.")

# Sanity checks at sample points (floating point, not part of the argument).
x = sp.Symbol("x", positive=True)
e_of_x = e.subs(E, sp.exp(x)).subs(X, x)
I = sp.sinh(6 * x) / 192 - sp.sinh(4 * x) / 64 - sp.sinh(2 * x) / 64 + x / 16
ode_residual = sp.diff(e_of_x, x) - (sp.tanh(x) * e_of_x - 3 * I / (sp.cosh(x) * sp.sinh(x) ** 5))
t = sp.Symbol("t", positive=True)
quadrature = 3 * sp.cosh(3) * sp.Integral((I / (sp.cosh(x) ** 2 * sp.sinh(x) ** 5)).subs(x, t), (t, 3, sp.oo))
print("Sanity check, ODE (2.9) residual at X = 3:", sp.N(ode_residual.subs(x, 3), 5))
print("Sanity check, series minus quadrature at X = 3:", sp.N(e_of_x.subs(x, 3) - quadrature, 5))
