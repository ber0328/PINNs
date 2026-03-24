# Hard Dirichlet Darcy — Half-Square Permeability: Experiment Summary

**Date:** March 16, 2026

---

## TL;DR

- We solve mixed-form Darcy flow with a sharp half-square jump in permeability ($k=10$ for $x_1<0$, $k=1$ for $x_1\ge 0$) using hard-enforced boundary conditions.
- Three models are compared: expert-gated (`combined_exp`), discontinuous-activation (`combined_disc`), and smooth baseline (`combined_smooth`).
- The expert model gives the best PDE residuals near the interface, but does not best match FEM in pressure error.
- The smooth baseline achieves the strongest FEM pressure accuracy in this setup.
- Main takeaway: for this half-square case, lower training residuals do not automatically imply best FEM agreement, and pherhaps interface/contact behavior remains the key challenge.

---

## 1. Problem Statement

We solve the mixed-form Darcy flow problem on the unit square $\Omega = [-1,1]^2$ with a **half-square** piecewise-constant permeability $k$:

$$\begin{align}
-k\,\nabla u &= v && \text{in } \Omega \\
\nabla \cdot v &= 0 && \text{in } \Omega \\
u &= g_D && \text{on } \Gamma_D \\
v &= h && \text{on } \Gamma_N
\end{align}$$

### Domain and permeability

The permeability field is discontinuous across the vertical line $x_1 = 0$:

$$k(x) = \begin{cases} 10 & x_1 < 0 \\ 1 & x_1 \geq 0 \end{cases}$$

The domain boundary is split into:
- **Dirichlet** $\Gamma_D$: the left and right edges ($x_1 = \pm 1$), where $u$ is prescribed.
- **Neumann** $\Gamma_N$: the top and bottom edges ($x_2 = \pm 1$), where the flux $v$ is prescribed.

### Boundary data

$$g_D(x) = \tfrac{1}{2}(x_1 + 1), \qquad h(x) = (0,\; x_2)^T$$

The Dirichlet condition imposes $u = 0$ on the left wall and $u = 1$ on the right wall.
The exact Neumann condition $h = (0, x_2)^T$ means that the y component of $v$ equals $\pm 1$ on the top/bottom faces.

---

## 2. Hard Enforcement of Boundary Conditions

All models use **hard enforcement**: boundary conditions are built directly into the network output
instead of being added as penalty terms in the loss.
This guarantees *exact* satisfaction of all prescribed boundary values for any network weights.

### General scheme

Given the raw network output $\hat{u}_\theta(x)$, the final prediction is:

$$u_\theta(x) = g_0(x) + g_1(x) \cdot \hat{u}_\theta(x)$$

where:

| Symbol | Requirement | Role |
|---|---|---|
| $g_0(x)$ | $g_0\big\|_\Gamma = g_\mathrm{BC}$ | **Lift** — already satisfies the BC |
| $g_1(x)$ | $g_1\big\|_\Gamma = 0$ | **Mask** — vanishes on the boundary |

Because $g_1 = 0$ on $\Gamma$, the learnable correction $g_1 \cdot \hat{u}_\theta$ has no effect there, and $u_\theta\big|_\Gamma = g_0\big|_\Gamma = g_\mathrm{BC}$ holds exactly.

### Functions used

#### Pressure field $u$ — Dirichlet BC $u\big|_{x_1=\pm1} = g_D$

| Name | Expression | Role |
|---|---|---|
| `g_D(x)` | $\frac{1}{2}(x_1 + 1)$ | Lift: equals $0$ on $x_1=-1$, equals $1$ on $x_1=+1$ |
| `u_0(x)` | $1 - x_1^2$ | Mask: zero on $x_1 = \pm 1$, positive in the interior |

$$u_\theta(x) = g_D(x) + u_0(x)\cdot\hat{u}_\theta(x)$$

#### Flux field $v$ — Neumann BC $v_2\big|_{x_2=\pm1} = x_2$

| Name | Expression | Role |
|---|---|---|
| `h(x)` | $(0,\; x_2)^T$ | Lift: matches the Neumann data on the top/bottom edges |
| `v_0(x)` | $(1,\; 1-x_2^2)^T$ | Mask: the $x_2$-component is zero on $x_2=\pm 1$; the $x_1$-component is 1 (no constraint on left/right walls for $v$) |

$$v_\theta(x) = h(x) + v_0(x) \odot \hat{v}_\theta(x)$$

---

## 3. Model Architectures

Three models are compared.
All share the same pressure sub-network and differ only in how the flux sub-network handles the interior discontinuity.

### `combined_exp` — Per-characteristic expert gating

The flux sub-network is a `PerCharModel` composed of two independent expert MLPs, each specialised for one subdomain and blended via the characteristic function:

$$\hat{v}_\theta(x) = \bigl(1-\chi(x)\bigr)\cdot\mathrm{Expert}_1(x) + \chi(x)\cdot\mathrm{Expert}_2(x)$$

Each expert is a 3-hidden-layer MLP with 16 units and `tanh` activations.
The weighted sum allows the network to represent a sharp discontinuity aligned with the interface line, with no cross-contamination between subdomains.

### `combined_disc` — Discontinuous last-layer activation

A standard `MLPModel` whose **last hidden layer** uses a `DiscTanh` activation — a learnable approximation of a Heaviside-type step function — in place of plain `tanh`.
This gives the network intrinsic capacity to learn sharp transitions without explicit domain decomposition.

### `combined_smooth` — Baseline smooth MLP

A plain `MLPModel` with 3 hidden layers of 32 units and `tanh` activations throughout.
No special treatment of the discontinuity; serves as the baseline.

### Architecture summary

| Model | Pressure net $u$ | Flux net $v$ | Discontinuity mechanism |
|---|---|---|---|
| `combined_exp` | MLP [32×3], tanh | Per-char experts [16×3 each] | Characteristic gating via $\chi$ |
| `combined_disc` | MLP [32×3], tanh | MLP [32×3], DiscTanh last layer | Learnable step-function activation |
| `combined_smooth` | MLP [32×3], tanh | MLP [32×3], tanh | None (smooth baseline) |

All models are trained for **120 000 Adam steps** (lr = 5×10⁻⁴, `ReduceLROnPlateau`, factor 0.75, patience 2000), then **1 000 L-BFGS steps** (lr = 0.1), using **10 000 Latin-hypercube collocation points** interior to $\Omega$.

---

## 4. Loss Function

The loss enforces the two PDE equations at the collocation points:

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum_{i=1}^N \bigl\|k(x_i)\nabla u_\theta(x_i) + v_\theta(x_i)\bigr\|_2^2}_{\mathcal{L}_1 \;:\; \text{Darcy law} \;(-k\nabla u = v)} \;+\; \underbrace{\frac{1}{N}\sum_{i=1}^N \bigl(\nabla \cdot v_\theta(x_i)\bigr)^2}_{\mathcal{L}_2 \;:\; \text{mass conservation} \;(\nabla\cdot v=0)}$$

Boundary conditions do **not** appear in the loss — they are satisfied exactly by construction.

---

## 5. Visualisations

### Predicted pressure $u_\theta(x)$

Contour plots for all three models.
The exact solution varies smoothly from $0$ (left wall, $x_1=-1$) to $1$ (right wall, $x_1=+1$), but with a kink in the gradient at $x_1=0$ due to the 10× permeability jump.

| Expert | Disc | Smooth |
|:---:|:---:|:---:|
| ![u_exp prediction](../images_hs/model_u_exp_pred.png) | ![u_disc prediction](../images_hs/model_u_disc_pred.png) | ![u_smooth prediction](../images_hs/model_u_smooth_pred.png) |

---

### Laplacian residual $|u_{xx} + u_{yy}|$

Pointwise absolute Laplacian, clamped to $[0, 0.1]$ (`combined_exp`) and $[0, 0.3]$ (others).
Elevated residual near $x_1 = 0$ reflects difficulty representing the gradient kink induced by the permeability jump.

| Expert | Disc | Smooth |
|:---:|:---:|:---:|
| ![u_exp laplacian](../images_hs/model_u_exp_laplacian.png) | ![u_disc laplacian](../images_hs/model_u_disc_laplacian.png) | ![u_smooth laplacian](../images_hs/model_u_smooth_laplacian.png) |

---

### Gradient field $\nabla u_\theta$

$30\times30$ arrow plots showing the heat-flux direction.
The magnitude should be smaller in the left (higher-$k$) region, with a discontinuous jump at the interface.

| Expert | Disc | Smooth |
|:---:|:---:|:---:|
| ![u_exp gradient](../images_hs/model_u_exp_grad.png) | ![u_disc gradient](../images_hs/model_u_disc_grad.png) | ![u_smooth gradient](../images_hs/model_u_smooth_grad.png) |

---

### Flux field $v_\theta(x)$

Vector field of the mixed-formulation flux.
The correct flux satisfies $v = -k\nabla u$, so its magnitude jumps by a factor of 10 across the interface at $x_1=0$.

| Expert | Disc | Smooth |
|:---:|:---:|:---:|
| ![v_exp](../images_hs/model_v_exp.png) | ![v_disc](../images_hs/model_v_disc.png) | ![v_smooth](../images_hs/model_v_smooth.png) |

---

### Pointwise Darcy-law residual $\|k\nabla u + v\|_2$

Contour plots (capped at 0.01) highlighting where $\mathcal{L}_1$ is largest.
The expert model shows the lowest errors near the interface.

| Expert | Disc | Smooth |
|:---:|:---:|:---:|
| ![v_exp div residual](../images_hs/model_v_exp_div.png) | ![v_disc div residual](../images_hs/model_v_disc_div.png) | ![v_smooth div residual](../images_hs/model_v_smooth_div.png) |

---

### Training convergence curves

Separate $\mathcal{L}_1$ and $\mathcal{L}_2$ vs. epoch plots for each model.

| Expert | Disc | Smooth |
|:---:|:---:|:---:|
| ![exp losses](../images_hs/model_exp_losses.png) | ![disc losses](../images_hs/model_disc_losses.png) | ![smooth losses](../images_hs/model_smooth_losses.png) |

---

### FEM error maps

After training, models are evaluated against the FEM reference solution in `darcy_half_square_solution.h5`:
- Row 1: $|u_\theta - u_\text{FEM}|$ — pressure absolute error.
- Row 2: $\|v_\theta - v_\text{FEM}\|_2$ — flux pointwise error magnitude.

![FEM error maps](../images_hs/model_errors_uv.png)

Moreover, the same comparison is made for components of the flux field against the FEM solution:
- Row 1: $|vx_\theta - vx_\text{FEM}|$ — $v$ x-component absolute error.
- Row 2: $|vy_\theta - vy_\text{FEM}|$ — $v$ y-component absolute error.

---

## 6. Quantitative Results

### L2 residual norms (PDE satisfaction, evaluated on a $50\times50$ grid)

| Model | $\|\mathcal{R}_1\|_{L^2}$ (Darcy law) | $\|\mathcal{R}_2\|_{L^2}$ (div-free) |
|---|---:|---:|
| `combined_exp` | 4.218277e-02 | 1.024649e-02 |
| `combined_disc` | 8.197301e-02 | 2.334311e-02 |
| `combined_smooth` | 9.050866e-02 | 1.888883e-02 |

### Comparison vs FEM reference solution

| Model | $u$ L2 err | $u$ Rel. L2 | $u$ RMSE | $v$ L2 err | $v$ Rel. L2 | $v$ RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `combined_exp` | 1.6089e+01 | 4.3713e-01 | 1.6089e-01 | — | — | — |
| `combined_disc` | 3.4403e+00 | 9.3471e-02 | 3.4403e-02 | — | — | — |
| `combined_smooth` | 1.4039e-01 | 3.8144e-03 | 1.4039e-03 | — | — | — |

*(Complete table produced by running the FEM comparison cell after training.)*

### Flux continuity condition at interface $x_1 = 0$

The physical transmission condition requires $v(x_1=0^-) = (k_\text{left}/k_\text{right})\,v(x_1=0^+) = 10\,v(x_1=0^+)$.
Models are evaluated at $x_1 = \pm\varepsilon$ ($\varepsilon = 10^{-4}$) along 400 sample points $x_2 \in (-1,1)$.

| Model | $v_x$ mean $\|\Delta\|$ | $v_x$ max $\|\Delta\|$ | $v_y$ mean $\|\Delta\|$ | $v_y$ max $\|\Delta\|$ |
|---|---:|---:|---:|---:|
| `combined_exp` | 1.6851e+00 | 1.7817e+01 | 6.1415e-01 | 8.8467e+00 |
| `combined_disc` | 1.4848e+00 | 6.2846e+00 | 2.8682e+00 | 8.9695e+00 |
| `combined_smooth` | 5.3772e+00 | 7.1992e+00 | 4.3488e+00 | 9.0171e+00 |
| **FEM reference** | **5.3453e+00** | **7.3737e+00** | **4.0574e+00** | **9.0105e+00** |

![ratio $v_1/v_2$ at $x=0$](../images_hs/model_flux_ratio_contact.png)

---

## 7. Discussion

- **Expert model (`combined_exp`) achieves the best Darcy-law residual** — however still underperforms, seemingly due to inability to capture proper contact condition, namely conservation of normal flux.
- **Smooth baseline (`combined_smooth`) achieves the best FEM pressure accuracy** (Rel. L2 = 0.38%). A simpler, smoother loss landscape benefits Adam and L-BFGS convergence more than any special discontinuity handling.
- **Largest parameter drift** in the expert model is consistent with its higher expressive capacity.
