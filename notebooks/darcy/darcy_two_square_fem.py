"""
darcy_two_square_fem.py
=======================
FEM reference solver for the *two-touching-squares* Darcy problem solved by
the PINN in ``darcy_learned_pou_difficulter.ipynb``.

Problem (mixed form, primal solved here)
-----------------------------------------
    -∇·(k(x,y) ∇u) = 0       in Ω = [-1,1]²
    u  = g_D = 0.5(x+1)       on Γ_D  (x = ±1)
    k ∂_n u = -1               on Γ_N  (y = ±1)

Piecewise-constant permeability over three regions:

    k = 1   –0.4 ≤ x ≤ 0,   |y| ≤ 0.4   (left  square)
    k = 10   0   ≤ x ≤ 0.4, |y| ≤ 0.4   (right square)
    k = 3   everywhere else               (background)

The Neumann value k ∂_n u = -1 is derived from the PINN Neumann data
h(x) = (0, y):  on Γ_N  v·n = h·n  →  (-k∂_y u)·(±1) = (±y)|_{y=±1} = 1
→  k ∂_y u = -1  on both top (n=(0,1)) and bottom (n=(0,-1)); equivalently
k ∂_n u = -1 everywhere on Γ_N.

Output
------
Saves ``darcy_two_square_solution.h5`` (same group/dataset layout as the
existing ``darcy_solution.h5``) containing:

    samples/xy           (N,2)  – sample grid coordinates
    samples/u_interp     (N,)   – interpolated pressure
    samples/flux_interp  (N,2)  – Darcy flux v = -k ∇u
    samples/kappa        (N,)   – permeability at sample points
    fem/nodes            (M,2)
    fem/elements         (M,3)
    fem/u                (M,)
    metadata/            – problem attributes

Mesh
----
250×250 uniform triangular mesh: mesh spacing 0.008, so the interfaces at
x = -0.4, 0, 0.4 and y = ±0.4 fall exactly on nodes.

Usage
-----
    conda run -n torch python darcy_two_square_fem.py
or simply:
    python darcy_two_square_fem.py          (with scipy, h5py, numpy, matplotlib)
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import LinearNDInterpolator

# ── Domain / physics parameters ──────────────────────────────────────────────
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0

# Mesh: 250 subdivisions → spacing 0.008; interfaces at ±0.4 and 0 land on
# exact mesh lines (0.4 / 0.008 = 50, i.e. integer multiples).
NX = NY = 250

# Permeability values
K_LEFT   = 1.0    # left  inner square
K_RIGHT  = 10.0   # right inner square
K_BG     = 3.0    # exterior background

# Inner-square geometry
SQ_X_MIN, SQ_X_MID, SQ_X_MAX =  -0.4, 0.0, 0.4
SQ_Y_ABS                       =  0.4

# Boundary conditions
V_LEFT, V_RIGHT = 0.0, 1.0              # Dirichlet: u at x=-1 and x=+1
NEUMANN_FLUX    = -1.0                  # k ∂_n u on top/bottom edges

# Output
OUT_PATH   = Path(__file__).parent / "darcy_two_square_solution.h5"
N_SAMPLES  = 100   # grid points per side → N_SAMPLES² total query points


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mesh generation (structured P1 triangles)
# ─────────────────────────────────────────────────────────────────────────────

def create_mesh(nx: int, ny: int,
                x_min: float = -1.0, x_max: float = 1.0,
                y_min: float = -1.0, y_max: float = 1.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (nodes (N_nodes,2), elems (N_elems,3)) for a structured P1 mesh
    on [x_min,x_max]×[y_min,y_max] with nx×ny quadrilaterals split diagonally
    into two triangles each."""
    x = np.linspace(x_min, x_max, nx + 1)
    y = np.linspace(y_min, y_max, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])   # row-major (y outer)

    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    elems: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            # Lower-left triangle
            elems.append([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1)])
            # Upper-right triangle
            elems.append([nid(i, j), nid(i + 1, j + 1), nid(i, j + 1)])

    return nodes, np.array(elems, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Permeability field
# ─────────────────────────────────────────────────────────────────────────────

def kappa_two_squares(xy: np.ndarray) -> np.ndarray:
    """Evaluate piecewise-constant k at an array of points (M, 2).

    Returns an array of shape (M,).
    """
    xy = np.atleast_2d(xy)
    x, y = xy[:, 0], xy[:, 1]
    in_left  = (x >= SQ_X_MIN) & (x <= SQ_X_MID) & (np.abs(y) <= SQ_Y_ABS)
    in_right = (x >= SQ_X_MID) & (x <= SQ_X_MAX) & (np.abs(y) <= SQ_Y_ABS)
    vals = np.full(len(xy), K_BG, dtype=float)
    vals[in_left]  = K_LEFT
    vals[in_right] = K_RIGHT
    return vals


def kappa_at_centroid(nodes: np.ndarray, elem: np.ndarray) -> float:
    """Permeability at the centroid of one element (1-point quadrature)."""
    centroid = nodes[elem].mean(axis=0)
    return float(kappa_two_squares(centroid[None])[0])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Element helpers
# ─────────────────────────────────────────────────────────────────────────────

def elem_grad_area(nodes: np.ndarray,
                   elem: np.ndarray) -> tuple[np.ndarray, float]:
    """Constant P1 gradients and (positive) area for one triangle."""
    (x0, y0), (x1, y1), (x2, y2) = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)   # twice signed area
    dN = np.array([
        [y1 - y2, x2 - x1],
        [y2 - y0, x0 - x2],
        [y0 - y1, x1 - x0],
    ], dtype=float) / area2          # shape (3, 2)
    return dN, abs(area2) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stiffness matrix assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_stiffness(nodes: np.ndarray,
                       elems: np.ndarray) -> sp.csr_matrix:
    """Assemble  K  for  a(u,v) = ∫ k(x) ∇u·∇v  dΩ  (1-point centroid quad)."""
    N = len(nodes)
    rows, cols, data = [], [], []
    for elem in elems:
        dN, area = elem_grad_area(nodes, elem)
        k_val = kappa_at_centroid(nodes, elem)
        K_loc = k_val * area * (dN @ dN.T)    # (3, 3)
        for a in range(3):
            for b in range(3):
                rows.append(elem[a])
                cols.append(elem[b])
                data.append(K_loc[a, b])
    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

def get_boundary_edges(elems: np.ndarray) -> list[tuple[int, int]]:
    """Return edges that appear in exactly one element (= boundary edges)."""
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for elem in elems:
        for i in range(3):
            e = tuple(sorted([int(elem[i]), int(elem[(i + 1) % 3])]))
            edge_count[e] += 1          # type: ignore[index]
    return [e for e, cnt in edge_count.items() if cnt == 1]


def apply_neumann(f: np.ndarray,
                  nodes: np.ndarray,
                  bnd_edges: list[tuple[int, int]],
                  flux: float,
                  y_min: float = Y_MIN,
                  y_max: float = Y_MAX,
                  tol: float = 1e-12) -> None:
    """Add  ∫_{Γ_N} flux · v  ds  to the load vector f (in-place).

    Only edges on y = y_min or y = y_max are treated as Γ_N.
    flux = k ∂_n u  on those edges.
    """
    for (i, j) in bnd_edges:
        pi, pj = nodes[i], nodes[j]
        mid_y  = 0.5 * (pi[1] + pj[1])
        if not (np.isclose(mid_y, y_min, atol=tol) or
                np.isclose(mid_y, y_max, atol=tol)):
            continue
        edge_len = np.linalg.norm(pj - pi)
        f[i] += flux * edge_len * 0.5
        f[j] += flux * edge_len * 0.5


def apply_dirichlet(K: sp.csr_matrix,
                    f: np.ndarray,
                    nodes: np.ndarray,
                    v_left: float = V_LEFT,
                    v_right: float = V_RIGHT,
                    x_min: float = X_MIN,
                    x_max: float = X_MAX,
                    tol: float = 1e-12):
    """Enforce u = 0.5*(x+1) on Γ_D (x=±1) by row/column elimination."""
    dir_vals: dict[int, float] = {}
    for i, (x, _) in enumerate(nodes):
        if np.isclose(x, x_min, atol=tol):
            dir_vals[i] = v_left                      # u = 0 at x = -1
        elif np.isclose(x, x_max, atol=tol):
            dir_vals[i] = v_right                     # u = 1 at x = +1

    K_lil = K.tolil()
    for i, val in dir_vals.items():
        col = K_lil.getcol(i).toarray().ravel()
        f  -= col * val       # move known columns to RHS
        K_lil[i, :] = 0.0
        K_lil[:, i] = 0.0
        K_lil[i, i] = 1.0
        f[i]        = val

    return K_lil.tocsr(), f, dir_vals


# ─────────────────────────────────────────────────────────────────────────────
# 6. Flux computation  v = -k ∇u  (P1: constant per element)
# ─────────────────────────────────────────────────────────────────────────────

def compute_flux(nodes: np.ndarray,
                 elems: np.ndarray,
                 u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids (M,2), flux (M,2)) where flux = -k ∇u per element."""
    centroids = nodes[elems].mean(axis=1)
    flux      = np.empty((len(elems), 2), dtype=float)
    for idx, elem in enumerate(elems):
        dN, _ = elem_grad_area(nodes, elem)
        grad_u     = dN.T @ u[elem]                      # (2,)
        k_val      = kappa_at_centroid(nodes, elem)
        flux[idx]  = -k_val * grad_u
    return centroids, flux


# ─────────────────────────────────────────────────────────────────────────────
# 7. Residual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_residuals(K: sp.csr_matrix,
                    f: np.ndarray,
                    u: np.ndarray,
                    nodes: np.ndarray,
                    dir_vals: dict[int, float],
                    tol: float = 1e-12) -> None:
    r_norm = np.linalg.norm(K @ u - f) / (np.linalg.norm(f) + 1e-300)
    print(f"  Algebraic relative residual : {r_norm:.3e}")
    assert r_norm < 1e-8, f"Solver did not converge (r={r_norm:.3e})"

    dir_err = max(abs(u[i] - v) for i, v in dir_vals.items())
    print(f"  Max Dirichlet BC error       : {dir_err:.3e}")
    assert dir_err < 1e-10, f"Dirichlet BC violated (err={dir_err:.3e})"
    print("  ✓ Residual checks passed.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_solution(nodes: np.ndarray,
                  elems: np.ndarray,
                  u: np.ndarray,
                  centroids: np.ndarray,
                  flux: np.ndarray,
                  save_path: Path | None = None) -> None:
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)
    mag = np.linalg.norm(flux, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # — pressure field
    ax = axes[0]
    tcf = ax.tricontourf(tri, u, levels=60, cmap='RdYlBu_r')
    fig.colorbar(tcf, ax=ax, label='pressure u')
    for x_iface in [SQ_X_MIN, SQ_X_MID, SQ_X_MAX]:
        ax.axvline(x_iface, color='white', lw=1.2, ls='--')
    ax.set_aspect('equal')
    ax.set_title('Pressure $u$')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    # — permeability
    ax2 = axes[1]
    k_elem = kappa_two_squares(centroids)
    tcf2 = ax2.tripcolor(tri, facecolors=k_elem, cmap='jet',
                         vmin=0, vmax=K_RIGHT)
    fig.colorbar(tcf2, ax=ax2, label='k(x,y)')
    ax2.set_aspect('equal')
    ax2.set_title('Permeability $k$')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')

    # — flux magnitude + quiver (subsampled)
    ax3 = axes[2]
    tcf3 = ax3.tripcolor(tri, facecolors=mag, cmap='magma')
    fig.colorbar(tcf3, ax=ax3, label=r'$\|v\|$')
    stride = max(1, len(centroids) // 1_200)
    c, fq = centroids[::stride], flux[::stride]
    ax3.quiver(c[:, 0], c[:, 1], fq[:, 0], fq[:, 1],
               color='white', alpha=0.6, scale=None, width=0.003)
    ax3.set_aspect('equal')
    ax3.set_title(r'Flux magnitude $\|v\|$ + arrows')
    ax3.set_xlabel('x'); ax3.set_ylabel('y')

    plt.suptitle('Two-square Darcy FEM  '
                 r'($k_L=1,\,k_R=10,\,k_{bg}=3$)', fontsize=13)
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9. HDF5 export
# ─────────────────────────────────────────────────────────────────────────────

def export_hdf5(out_path: Path,
                nodes: np.ndarray,
                elems: np.ndarray,
                u: np.ndarray,
                sample_pts: np.ndarray,
                u_sampled: np.ndarray,
                flux_sampled: np.ndarray,
                k_sampled: np.ndarray,
                n_side: int) -> None:
    """Write results to HDF5, matching the layout of ``darcy_solution.h5``."""
    with h5py.File(out_path, 'w') as f:
        # ── metadata ──────────────────────────────────────────────────────────
        meta = f.create_group('metadata')
        meta.attrs['domain']        = f'[{X_MIN},{X_MAX}]x[{Y_MIN},{Y_MAX}]'
        meta.attrs['nx']            = NX
        meta.attrs['ny']            = NY
        meta.attrs['n_fem_nodes']   = len(nodes)
        meta.attrs['n_fem_elems']   = len(elems)
        meta.attrs['k_left_sq']     = K_LEFT
        meta.attrs['k_right_sq']    = K_RIGHT
        meta.attrs['k_background']  = K_BG
        meta.attrs['sq_x_min']      = SQ_X_MIN
        meta.attrs['sq_x_mid']      = SQ_X_MID
        meta.attrs['sq_x_max']      = SQ_X_MAX
        meta.attrs['sq_y_abs']      = SQ_Y_ABS
        meta.attrs['v_left']        = V_LEFT
        meta.attrs['v_right']       = V_RIGHT
        meta.attrs['neumann_flux']  = NEUMANN_FLUX

        # ── FEM mesh + nodal solution ──────────────────────────────────────────
        fem = f.create_group('fem')
        fem.create_dataset('nodes',    data=nodes,  compression='gzip')
        fem.create_dataset('elements', data=elems,  compression='gzip')
        fem.create_dataset('u',        data=u,      compression='gzip')

        # ── sampled data (used directly by the PINN notebook) ─────────────────
        samp = f.create_group('samples')
        samp.attrs['n_points']  = len(sample_pts)
        samp.attrs['grid_size'] = n_side
        samp.create_dataset('xy',          data=sample_pts,  compression='gzip')
        samp.create_dataset('u_interp',    data=u_sampled,   compression='gzip')
        samp.create_dataset('flux_interp', data=flux_sampled, compression='gzip')
        samp.create_dataset('kappa',       data=k_sampled,   compression='gzip')

    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── (a) Build mesh ────────────────────────────────────────────────────────
    print(f"Building {NX}×{NY} mesh …")
    nodes, elems = create_mesh(NX, NY, X_MIN, X_MAX, Y_MIN, Y_MAX)
    print(f"  {len(nodes)} nodes,  {len(elems)} elements")

    # ── (b) Assemble stiffness matrix ─────────────────────────────────────────
    print("Assembling stiffness matrix …")
    K = assemble_stiffness(nodes, elems)

    # ── (c) Build load vector & Neumann BCs ───────────────────────────────────
    f = np.zeros(len(nodes))
    bnd_edges = get_boundary_edges(elems)
    apply_neumann(f, nodes, bnd_edges, flux=NEUMANN_FLUX)

    # ── (d) Dirichlet BCs ─────────────────────────────────────────────────────
    K, f, dir_vals = apply_dirichlet(K, f, nodes)

    # ── (e) Solve K u = f ─────────────────────────────────────────────────────
    print("Solving linear system …")
    u = spla.spsolve(K, f)
    print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"  Elapsed: {time.time()-t0:.1f} s")

    # ── (f) Residual checks ───────────────────────────────────────────────────
    print("Checking residuals …")
    check_residuals(K, f, u, nodes, dir_vals)

    # ── (g) Darcy flux v = -k ∇u ─────────────────────────────────────────────
    print("Computing Darcy flux …")
    centroids, flux = compute_flux(nodes, elems, u)
    print(f"  |v| range: [{np.linalg.norm(flux, axis=1).min():.4f}, "
          f"{np.linalg.norm(flux, axis=1).max():.4f}]")

    # ── (h) Sample onto regular grid ─────────────────────────────────────────
    print(f"Sampling onto {N_SAMPLES}×{N_SAMPLES} grid …")
    xs = np.linspace(X_MIN, X_MAX, N_SAMPLES)
    ys = np.linspace(Y_MIN, Y_MAX, N_SAMPLES)
    gx, gy = np.meshgrid(xs, ys)
    sample_pts = np.column_stack([gx.ravel(), gy.ravel()])   # (N², 2)

    interp_u  = LinearNDInterpolator(nodes, u)
    interp_vx = LinearNDInterpolator(centroids, flux[:, 0])
    interp_vy = LinearNDInterpolator(centroids, flux[:, 1])

    u_sampled    = interp_u(sample_pts)
    flux_sampled = np.column_stack([interp_vx(sample_pts),
                                    interp_vy(sample_pts)])
    k_sampled    = kappa_two_squares(sample_pts)

    n_valid = int(np.sum(~np.isnan(u_sampled)))
    print(f"  Valid interpolated values: {n_valid}/{len(sample_pts)}")
    print(f"  u_sampled range: [{np.nanmin(u_sampled):.4f}, {np.nanmax(u_sampled):.4f}]")
    print(f"  k values: {np.unique(k_sampled)}")

    # ── (i) Save HDF5 ────────────────────────────────────────────────────────
    print(f"Writing {OUT_PATH} …")
    export_hdf5(OUT_PATH, nodes, elems, u,
                sample_pts, u_sampled, flux_sampled, k_sampled,
                n_side=N_SAMPLES)

    # ── (j) Verify by reading back ───────────────────────────────────────────
    print("Verifying …")
    with h5py.File(OUT_PATH, 'r') as f:
        xy_rb = f['samples/xy'][:]
        u_rb  = f['samples/u_interp'][:]
        v_rb  = f['samples/flux_interp'][:]
        k_rb  = f['samples/kappa'][:]
    print(f"  Read back: xy {xy_rb.shape},  u {u_rb.shape},  "
          f"v {v_rb.shape},  k {k_rb.shape}")

    # ── (k) Plot ─────────────────────────────────────────────────────────────
    print("Plotting …")
    plot_path = OUT_PATH.with_suffix('.png')
    plot_solution(nodes, elems, u, centroids, flux, save_path=plot_path)

    print(f"\nDone in {time.time()-t0:.1f} s.")


if __name__ == '__main__':
    main()
