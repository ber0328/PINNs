# Hard Dirichlet Darcy — Summary Report

**$k$:** $1$ inside $\|x\|<0.5$, $[\cdot]$ outside

---

## 1) Models

| Model | Pressure net | Flux net | Discontinuity mechanism |
|---|---|---|---|
| `combined_exp` | MLP [32×3] | Per-char experts [16×3 each] | Characteristic gating ($\chi$) |
| `combined_disc` | MLP [32×3] | MLP [32×3] | `heaviside + tanh` last-layer activation |
| `combined_smooth` | MLP [32×3] | MLP [32×3] | None (baseline) |

All models hard-enforce $u|_{\Gamma_D}=g_D$ and $v\cdot n|_{\Gamma_N}=h$.

---

## 2) Results vs FEM ground truth

$$\mathrm{L2}=\int_{\Omega} \| u_{\text{model}} - u_{\text{FEM}} \|^2dV$$

| Model | $L_2$ error |
|---|---:|
| `smooth_u` | 3.920896e-03 |

---

## 3) Training convergence

| Model | res1 L2 (v = -k grad u) | res2 L2 (div v) |
|---|---:|---:|
| `combined_exp` | 3.475470e-02 | 2.990495e-02 |
| `combined_disc` | 1.767516e-01 | 3.396903e-02 |
| `combined_smooth` |1.105525e-01  | 2.764679e-02 |

---

## 4) Max parameter drift $\max|\Delta\theta|$

| Model | $\max\|\Delta\theta\|$ | $\\mean\|\Delta\theta\|$ |
|---|---|---|
| combined_exp | `3.116803e+00` | `1.624715e-01` | 
| combined_disc  | `2.784418e+00` | `8.288072e-02` |
| combined_smooth  | `1.929538e+00` | `1.303391e-01` |

---

## 5) Discussion

- The most suprising result is, that networks simply using heaviside or its variations performed worst of the models tested. 
- Another suprise was the final L2 erro of the smooth model against ground truth. Among many tests, the smooth model performed better than other models.
- So far, the domain decomposing model was a great dissapointement, in that it never learned the v-function properly, in that it did not respect contact boundary conditions.
- The effects of the last part are evident for high **k** differences (oter 1, inner 20)

