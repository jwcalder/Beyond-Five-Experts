# Beyond Five Experts

Verification scripts for the paper

> Jeff Calder and Nadejda Drenska, *On prediction from expert advice with more than five experts*, 2026.
> arXiv link: to be added.

The paper proves that for six or more experts, no single rank-ordered adversary strategy is globally optimal for prediction with expert advice, in both the geometric-stopping and finite-horizon settings. The proofs of the main theorems are written out in the paper and do not use the scripts in this repository. They do rely on results from the companion papers on four and five experts, which come with their own verification code.

The two scripts here check explicit computations stated in the paper:

| Script | Where it is used in the paper | What it checks | Run time |
|---|---|---|---|
| `five_expert_tail.py` | Remark 4.12 | The one-leader tail of the explicit five-expert solution has the coefficient predicted by Theorem 2.1 (exact rational arithmetic) | about 10 s |
| `origin_bounds.py` | Appendix A: Lemma A.1(ii), Lemma A.2(iii)–(iv), Remark A.3 | Comparisons between bounds on the value at the origin (interval arithmetic and exact rational arithmetic) | about 30 s |

## Requirements and usage

Python 3 with [SymPy](https://www.sympy.org) and [mpmath](https://mpmath.org). mpmath is installed automatically with SymPy.

```bash
pip install -r requirements.txt
python3 five_expert_tail.py
python3 origin_bounds.py
```

Each script stops with an `AssertionError` if any check fails. On success, `five_expert_tail.py` prints `Exact: F = (32/35) e^{-3a} + O(a e^{-5a}), ...`, and `origin_bounds.py` ends with `All certificates passed.`

## Notation

The paper normalizes the value function $u_n$ so that it solves

$$u_n - \tfrac12 \max_{v\in\{0,1\}^n} v^T D^2 u_n\, v = \max_i x_i, \qquad \kappa = \sqrt2 .$$

$u_n(0)$ is the value when all $n$ experts are tied. $Z_1,\dots,Z_n$ are independent standard normal random variables, and $X_n = \max_{i\le n} Z_i$.

| Symbol | Meaning | Source |
|---|---|---|
| $h_{n-1}(0) = \dfrac{(2n-2)!!}{4\kappa\,(2n-3)!!}$ | barrier bound at the origin | this paper, Proposition 4.5 |
| $B_n = \dfrac{n-1}{n\kappa}\Bigl(1+\log\dfrac{n\kappa h_{n-1}(0)}{n-1}\Bigr)$ | convexity bound, $n\ge4$ | this paper, (A.1) |
| $C_n = \sqrt{\pi\log n/8}$ | finite-horizon comparison bound | this paper, (A.1) |
| $H_n = \sqrt{\pi/8}\,\mathbb E X_n$ | heat-based player | Kobzar–Kohn–Wang, (A.2) |
| $W_n = \sqrt{(\log n)/2}$ | exponential-weights player | Kobzar–Kohn–Wang, (A.2) |
| $M_n$ | max-potential player: $\sqrt{(n-1)/8}$ ($n$ even), $\frac{n-1}{n}\sqrt{(n+1)/8}$ ($n$ odd) | Kobzar–Kohn–Wang, (A.2) |
| $L_n = \frac{\sqrt\pi}{4}\sqrt{2c_n}\,\mathbb E X_n$ | heat-based adversary (lower bound); $2c_n = n/(n-1)$ ($n$ even), $(n+1)/n$ ($n$ odd) | Kobzar–Kohn–Wang |
| $\ell_n = 10^{-3}\sum_{j=1}^{10^4} f_n(j/10^3)$ | certified lower bound for $\mathbb E X_n$, with $f_n = 1-\Phi^n-(1-\Phi)^n$ | this paper, Lemma A.1(i) |

## `five_expert_tail.py` (Remark 4.12)

Theorem 2.1 of the paper shows that when one expert leads by $s$,

$$u_n(s,0,\dots,0) - s \sim h_{n-1}(0)\, e^{-\sqrt2 s} \qquad (s\to\infty).$$

For five experts, $h_4(0) = 32/(35\sqrt2)$. The script checks this against the explicit five-expert solution of Calder and Drenska ([arXiv:2609.14892](https://arxiv.org/abs/2609.14892)). Along $x=(s,0,0,0,0)$ that solution is given in its Region III by formulas (2.16)–(2.17) there, with $a = \sqrt2 s/3$. The script shows, in exact rational arithmetic, that

$$F = \tfrac{32}{35}\, e^{-3a} + O(a e^{-5a}).$$

The script works as follows:

1. It expands the trace quadrature (2.8) of the companion paper in powers of $e^{-2X}$. This is a convergent series with coefficients affine in $X$, so it can be integrated and differentiated term by term.
2. It substitutes the series for $e, e', e'', e'''$ and the series $\lambda(a) = \log\coth(a/2) = 2\sum_j e^{-(2j+1)a}/(2j+1)$ into (2.16)–(2.17).
3. It asserts that every term of order at least $e^{-3a}$ cancels, except $\tfrac{32}{35}e^{-3a}$.

In (2.16)–(2.17), $\lambda$ is multiplied by a term of order $e^{4a}$, so it must be kept through $e^{-7a}$. The docstring records what goes wrong with fewer terms.

Two floating-point sanity checks at a sample point compare the series with the ODE (2.9) and with direct quadrature. They are not part of the verification.

This script is an independent cross-check of Theorem 2.1 against the five-expert formula. It is not used in any proof.

## `origin_bounds.py` (Appendix A)

Appendix A of the paper compares upper bounds on $u_n(0)$. Our two new bounds, $B_n$ and $C_n$, improve on the previous bounds $H_n$, $W_n$ and $M_n$ of Kobzar, Kohn and Wang. The smallest of the compared bounds is $B_n$ for $6\le n\le 13$ and $C_n$ for every $n\ge14$. The script has two parts.

**Part 1 (display only).** It prints the rows of the table in Remark A.3, in floating point, with $\mathbb E X_n$ computed by quadrature.

**Part 2 (certificates).** Every numerical inequality used in Appendix A is checked with mpmath's interval arithmetic (`mpmath.iv`, outward rounding), or in exact rational arithmetic.

- **Enclosure of $\Phi$.** The normal distribution function is enclosed at $x = j/1000$, $1\le j\le 10^4$, by the Taylor series of $\int_0^x e^{-s^2/2}\,ds$ together with the Lagrange remainder bound $x^{2K+3}/(2^{K+1}(K+1)!\,(2K+3))$, which is valid for every $x\ge0$.
- **Lower bounds for $\mathbb E X_n$.** $\mathbb E X_n = \int_0^\infty f_n$, and $f_n$ is nonnegative and decreasing on $[0,\infty)$. So the right Riemann sum $\ell_n$ is a lower bound for $\mathbb E X_n$ (Lemma A.1(i)).

The certified checks are:

- **(a) Lemma A.1(ii): $\mathbb E X_n > \sqrt{\log n}$, i.e. $C_n < H_n$, for $9\le n\le 2.5\times10^{13}$.** $\mathbb E X_n$ is nondecreasing in $n$, so a certified bound $\ell_m > \sqrt{\log N}$ settles every $n$ with $m \le n \le N$. Eleven such blocks, starting at $m=9$, cover the range. For $n \ge 10^{13}$ the paper gives an analytic argument, and the script checks the one numerical constant it uses.
- **(b) Lemma A.2(iii).** $L_6$, $L_7$ and $\frac{\sqrt\pi}{4}\ell_8$ all exceed the five-expert value $u_5(0) = 45\pi^2/(512\sqrt2)$. Monotonicity then gives $L_n > u_5(0)$ for every $n\ge6$.
- **(c) Lemma A.2(iv).** $B_{14} > C_{14}$, by exact rational arithmetic, using exponential-series bounds and $\pi < 22/7$. Combined with the paper's proof that $B_n/\sqrt{\log n}$ is increasing, this gives $B_n > C_n$ for all $n\ge14$.
- **(d) Remark A.3.**
  - $B_n < \min(H_n, W_n, M_n)$ for $4\le n\le 23$.
  - $B_n < C_n$ for $4\le n\le 13$.
  - The exact values for $n=4,5$ lie below every upper bound.
  - $C_{14}$ is the smallest bound at $n=14$.
  - The identities $\kappa u_2(0) = \kappa h_1(0) = 1/2$ and $\kappa u_3(0) = \kappa h_2(0) = 2/3$ hold exactly.

All comparisons for larger $n$ are proved analytically in the paper.

## Citation

```bibtex
@article{calder2026prediction,
  author  = {Calder, Jeff and Drenska, Nadejda},
  title   = {On prediction from expert advice with more than five experts},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
