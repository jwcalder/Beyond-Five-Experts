"""Bounds on u_n(0): table and certified comparisons (Appendix A).

With kappa = sqrt(2) and Z_1, ..., Z_n independent standard normals, the
bounds compared in Appendix A are

    h_{n-1}(0) = (2n-2)!! / (4 kappa (2n-3)!!)                  barrier (this paper),
    B_n = (n-1)/(n kappa) (1 + log(n kappa h_{n-1}(0)/(n-1)))   convexity (this paper),
    C_n = sqrt(pi log n / 8)                                    transform (this paper),
    H_n = sqrt(pi/8) E max_i Z_i                                heat-based player [KKW],
    W_n = sqrt((log n)/2)                                       exponential weights [KKW],
    M_n = sqrt((n-1)/8) (n even), (n-1)/n sqrt((n+1)/8) (n odd) max-potential player [KKW],
    L_n = (sqrt(pi)/4) sqrt(2 c_n) E max_i Z_i                  heat-based adversary [KKW],

with 2c_n = n/(n-1) for even n and (n+1)/n for odd n, and the exact values
u_n(0) for n <= 5. [KKW] is Kobzar, Kohn and Wang, kobzar2020newgeometric.

Part 1 prints the rows of the table in Remark A.3 in floating point, computing
E max_i Z_i by mpmath quadrature. These values are for display only.

Part 2 certifies, in interval arithmetic with outward rounding (mpmath.iv),
every inequality of Appendix A that involves a numerical value.
  - Phi(x) is enclosed at x = j/1000, 1 <= j <= 10^4, by the Taylor series of
    int_0^x exp(-s^2/2) ds with the Lagrange remainder bound
    x^(2K+3) / (2^(K+1) (K+1)! (2K+3)), valid for every x >= 0.
  - E max_{i<=n} Z_i = int_0^oo f_n with f_n = 1 - Phi^n - (1 - Phi)^n, which is
    nonnegative and decreasing on [0, oo), so the right Riemann sum
    ell_n = 10^-3 sum_{j=1}^{10^4} f_n(j/1000) is a lower bound (Lemma A.1(i)).
  (a) Lemma A.1(ii): E max_{i<=n} Z_i > sqrt(log n) for 9 <= n <= 10^13, in blocks:
      since E max is nondecreasing in n, ell_m > sqrt(log M) settles m <= n <= M.
      Also the constant used for n >= 10^13 in the proof.
  (b) Lemma A.2(iii): L_6, L_7 and (sqrt(pi)/4) ell_8 exceed u_5(0).
  (c) Lemma A.2(iv): B_14 > C_14, by exact rational arithmetic.
  (d) Remark A.3: the comparisons for n <= 23.

Run: python3 origin_bounds.py   (requires mpmath; takes about a minute)
"""

from fractions import Fraction
from math import factorial, floor

import mpmath as mp


def double_factorial_ratio(m):
    """(2m)!! / (2m-1)!! as an exact fraction."""
    r = Fraction(1)
    for j in range(1, m + 1):
        r *= Fraction(2 * j, 2 * j - 1)
    return r


def kappa_h0(n):
    """kappa h_{n-1}(0) as an exact fraction."""
    return double_factorial_ratio(n - 1) / 4


def two_c(n):
    return Fraction(n, n - 1) if n % 2 == 0 else Fraction(n + 1, n)


# Lemma A.2(ii): kappa h_{n-1}(0) has the values 4/5, 32/35, 64/63 at n = 4, 5, 6.
assert [kappa_h0(n) for n in (4, 5, 6)] == [Fraction(4, 5), Fraction(32, 35), Fraction(64, 63)]

# ----------------------------------------------------------------------------
# Part 1: the table of Remark A.3 (floating point, for display).
# ----------------------------------------------------------------------------
mp.mp.dps = 30
kappa = mp.sqrt(2)
fl = lambda q: mp.mpf(q.numerator) / q.denominator

h0 = lambda n: fl(kappa_h0(n)) / kappa
B = lambda n: (n - 1) / (n * kappa) * (1 + mp.log(n * fl(kappa_h0(n)) / (n - 1)))
C = lambda n: mp.sqrt(mp.pi * mp.log(n) / 8)
W = lambda n: mp.sqrt(mp.log(n) / 2)
M = lambda n: mp.sqrt(mp.mpf(n - 1) / 8) if n % 2 == 0 else mp.mpf(n - 1) / n * mp.sqrt(mp.mpf(n + 1) / 8)


def emax(n):
    """E max of n independent standard normals, by quadrature."""
    n = mp.mpf(n)
    above = lambda t: -mp.expm1(n * mp.log(mp.ncdf(t)))
    below = lambda t: mp.exp(n * mp.log(mp.ncdf(t)))
    c = mp.sqrt(2 * mp.log(n))
    nodes = [0] + [c * k / 8 for k in range(1, 13)] + [2 * c, mp.inf]
    return mp.quad(above, nodes) - mp.quad(below, [-mp.inf, -4, -1, 0])


H = lambda n: mp.sqrt(mp.pi / 8) * emax(n)
L = lambda n: mp.sqrt(mp.pi) / 4 * mp.sqrt(fl(two_c(n))) * emax(n)
exact = {4: mp.pi / (4 * mp.sqrt(2)), 5: 45 * mp.pi ** 2 / (512 * mp.sqrt(2))}

print("Table of Remark A.3 (smallest upper bound in bold):")
print("n & u_n(0) & L_n & h_{n-1}(0) & B_n & C_n & H_n & W_n & M_n")
for n in range(4, 15):
    uppers = {"h": h0(n), "B": B(n), "C": C(n), "H": H(n), "W": W(n), "M": M(n)}
    best = min(uppers, key=uppers.get)
    cell = lambda k: (r"\textbf{%.4f}" if k == best else "%.4f") % float(uppers[k])
    u = "%.4f" % float(exact[n]) if n in exact else "--"
    print(" & ".join([str(n), u, "%.4f" % float(L(n))] + [cell(k) for k in "hBCHWM"]) + r" \\")

# ----------------------------------------------------------------------------
# Part 2: certificates in interval arithmetic.
# ----------------------------------------------------------------------------
iv = mp.iv
iv.dps = 60
STEP = iv.mpf(1) / 1000
lt = lambda a, b: a.b < b.a  # certified a < b for intervals a, b
down = lambda x, d: f"{floor(float(x) * 10 ** d) / 10 ** d:.{d}f}"  # print a lower bound, rounded down
ival = lambda q: iv.mpf(q.numerator) / q.denominator


def phi_enclosures():
    """Intervals containing Phi(j/1000) for j = 1, ..., 10^4."""
    inv = 1 / iv.sqrt(2 * iv.pi)
    tol = iv.mpf(10) ** -45
    out = []
    for j in range(1, 10001):
        x = STEP * j
        x2 = x * x
        term, total, k = x, x, 0  # term = x (-x^2/2)^k / k!
        while True:
            k += 1
            term = term * (-x2) / (2 * k)
            total += term / (2 * k + 1)
            rem = abs(term * x2 / (2 * (k + 1))) / (2 * k + 3)  # Lagrange remainder bound
            if rem.b < tol.a:
                break
        out.append(iv.mpf("0.5") + inv * (total + iv.mpf([-rem.b, rem.b])))
    return out


PHI = phi_enclosures()


def ell(n):
    """Certified lower bound for E max_{i<=n} Z_i (a point interval)."""
    s = iv.mpf(0)
    for p in PHI:
        s += 1 - p ** n - (1 - p) ** n
    return (STEP * s).a


ikappa = iv.sqrt(2)
ih0 = lambda n: ival(kappa_h0(n)) / ikappa
iB = lambda n: iv.mpf(n - 1) / (n * ikappa) * (1 + iv.log(ival(kappa_h0(n) * Fraction(n, n - 1))))
iC = lambda n: iv.sqrt(iv.pi * iv.log(iv.mpf(n)) / 8)
iW = lambda n: iv.sqrt(iv.log(iv.mpf(n)) / 2)
iM = lambda n: iv.sqrt(iv.mpf(n - 1) / 8) if n % 2 == 0 else iv.mpf(n - 1) / n * iv.sqrt(iv.mpf(n + 1) / 8)
iH = lambda n: iv.sqrt(iv.pi / 8) * ell(n)  # lower endpoint is a lower bound for H_n
iL = lambda n: iv.sqrt(iv.pi) / 4 * iv.sqrt(ival(two_c(n))) * ell(n)  # likewise for L_n
iexact = {
    2: iv.sqrt(2) / 4,
    3: iv.sqrt(2) / 3,
    4: iv.pi / (4 * iv.sqrt(2)),
    5: 45 * iv.pi ** 2 / (512 * iv.sqrt(2)),
}

# (a) Lemma A.1(ii) for 9 <= n <= 10^13, in blocks.
print("\n(a) E max_{i<=n} Z_i > sqrt(log n):")
m, blocks = 9, 0
while m <= 10 ** 13:
    lo = ell(m)
    cap = iv.exp(lo ** 2).a  # every integer N < cap has log N < lo^2
    top = int(cap)
    if not iv.mpf(top) < cap:
        top -= 1
    assert top >= m, m
    print(f"    ell_{m} > {down(lo.a, 10)} gives {m} <= n <= {top}")
    m, blocks = top + 1, blocks + 1
print(f"    certified for 9 <= n <= {m - 1} in {blocks} blocks")
N13 = iv.mpf(10) ** 13
assert (iv.exp(iv.log(N13) / 4) / (2 * iv.sqrt(3 * iv.pi * iv.log(N13)))).a > 25

# (b) Lemma A.2(iii).
u5 = iexact[5]
margins = [iL(6) - u5, iL(7) - u5, iv.sqrt(iv.pi) / 4 * ell(8) - u5]
assert all(d.a > 0 for d in margins)
print("(b) L_6 - u_5(0), L_7 - u_5(0), (sqrt(pi)/4) ell_8 - u_5(0) exceed",
      ", ".join(down(d.a, 6) for d in margins))

# (c) Lemma A.2(iv): B_14 > C_14. With T = 14 kappa h_13(0)/13, log T > 69/125 and
# log 14 < 66/25 by exponential series bounds, and pi < 22/7.
T = kappa_h0(14) * Fraction(14, 13)
assert T == Fraction(4194304, 2414425)
x = Fraction(69, 125)
assert sum(x ** j / factorial(j) for j in range(11)) + x ** 11 / factorial(11) / (1 - x / 12) < T
assert sum(Fraction(66, 25) ** j / factorial(j) for j in range(16)) > 14
assert (Fraction(13, 14) * (1 + x)) ** 2 - Fraction(22, 7) * Fraction(66, 25) / 4 == Fraction(1996, 765625)
assert lt(iC(14), iB(14))
print("(c) (kappa B_14)^2 - (kappa C_14)^2 > 1996/765625")

# (d) Remark A.3.
# n = 2, 3: equality with the barrier, exactly: kappa u_2(0) = 2/4 and kappa u_3(0) = 2/3
# from the closed forms sqrt(2)/4 and sqrt(2)/3.
assert kappa_h0(2) == Fraction(2, 4) and kappa_h0(3) == Fraction(2, 3)
for n in (4, 5):  # exact values below every upper bound
    assert all(lt(iexact[n], b) for b in (ih0(n), iB(n), iC(n), iH(n), iW(n), iM(n)))
for n in range(4, 24):  # B_n below all three previous upper bounds
    assert lt(iB(n), iH(n)) and lt(iB(n), iW(n)) and lt(iB(n), iM(n))
for n in range(4, 14):  # B_n below C_n, so B_n is the smallest for n <= 13
    assert lt(iB(n), iC(n))
assert all(lt(iC(14), b) for b in (ih0(14), iB(14), iH(14), iW(14), iM(14)))
print("(d) comparisons of Remark A.3 certified for n <= 23")
print("All certificates passed.")
