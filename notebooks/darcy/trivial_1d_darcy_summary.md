# 1-D Darcy — Results Summary

**Problem:** Mixed-form Darcy on $[-1, 1]$ with piecewise-constant coefficient
$k(x) = 1$ for $x \in (-\tfrac{1}{3}, \tfrac{1}{3})$, $k(x) = 10$ otherwise.

**Boundary condition:** $u(\pm 1) = 1 \mp 1 = 0/2$ (hard-enforced via lifting).

---

## Residuals after training

| Model | $\|v + k\,u_x\|_{L^2}$ | $\|v_x\|_{L^2}$ |
|---|---|---|
| Expert (per-char) | `9.803084e-02` | `2.646687e-03` |
| Naive (MLP)       | `8.247618e-02` | `2.277828e-03` |

## Max parameter drift $\max|\Delta\theta|$

| Model | $\max\|\Delta\theta\|$ |
|---|---|
| Expert | `1.281461e+00` |
| Naive  | `1.790035e+00` |

---

## Observations

- The **expert model** captures the flux discontinuity at $x = \pm\tfrac{1}{3}$ by splitting $v$ into two characteristic-weighted sub-networks. The main point of this experiment was to determine, or rather sanity check, if such architecture even has a chance of converging to a real solution. To this end, one may consider this experiment a **succsess**. It is expected, that on harder problems, such as darcy in higher dimension, this architecture could pay off dividends.
- The **naive MLP** doesnt really struggle due to simplicity of the problem to represent the sharp jump in $v$, resulting in comparable residuals and similair parameter drift.

## Overall notes

- It seems that this architecture is capable of learning true solutions, without any major problems. The smoothness of u_model likely forces the v_model to remain reasonable, even if it essentially consists of separate functions on separate regions.
- This approach may be generalized to arbitrary regions, as the characteristic function for it may be approximated by neural.

## Further work

- Obviously, such aproach is to be tested on larger and more difficult problems.