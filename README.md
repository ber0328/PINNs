# PINNs Introduction — Summary Index

This README is a central index of all current markdown summaries (`.md`) in this repository.

Use this page to quickly find:
- what experiment each summary covers,
- when it was run (if recorded),
- where to add brief notes.

---

## Current Summary Files

| Summary file | Experiment date | Brief experiment description | Result |
|---|---|---|---|
| [notebooks/darcy/summaries/hard_dirichlet_summary.md](notebooks/darcy/summaries/hard_dirichlet_summary.md) | March 15, 2026 | Hard-Dirichlet Darcy on square domain with circular permeability interface; compares expert/disc/smooth PINNs and FEM errors. | Turns out mixture of experts likely struggles with contact conditions, as even though it performed better when it came to pure residual norm, it simply did not learn the true solution, as shown in comparison with FEM solution. |
| [notebooks/darcy/summaries/hard_dirichlet_hs_summary.md](notebooks/darcy/summaries/hard_dirichlet_hs_summary.md) | March 16, 2026 | Hard-Dirichlet Darcy with half-square discontinuity at x=0; includes transmission-condition analysis on the contact boundary. | Same results as above. |
| [notebooks/toy_problems/discontinuous_summary.md](notebooks/toy_problems/discontinuous_summary.md) | ?? | Discontinuous function approximation benchmarks (1-D and 2-D) across heaviside-based and expert-characteristic models. | Indeed, it "prooved" MoE has the best potential to express piecewise smooth functions, once the domain decomposition is known. |
| [notebooks/darcy/summaries/moe_xpinn_darcy_summary.md](notebooks/darcy/summaries/moe_xpinn_darcy_summary.md) | March 24, 2026 | Hard-Dirichlet Darcy on square domain with circular permeability interface; trying XPINNs + MoE. | This time, it worked. It turns out that it was the case that MoE trained each expert to minimise residuals on each, yet it had no incentive to learn the continuity of flux at material interface. Penalizing the violation of flux continuity actually helped with the training and yielded much better results than before. The challange is, how can we generalize this results for "un-nice" material environmets. |
| [notebooks/darcy/summaries/moe_xpinn_learned_pou.md](notebooks/darcy/summaries/moe_xpinn_learned_pou.md) | April ??, 2026 | XPINN+MoE with learned PoU | Training the networks without knowing the exact partition of unity. To find them, larger MLPs were trained against the known permeability to recognize each region. The normal vectors were found using automatic differentiation. Afterwards, we froze the chi-models gradients and moved on training as before.|
| [notebooks/darcy/summaries/hard_dirichlet_harderer_pou.md](notebooks/darcy/summaries/hard_dirichlet_harderer_pou.md) | April 7, 2026 | XPINN+MoE with learned PoU on two square domain | Essentially the same problem as before, except with more regions, sharper corners and bigger differences in k with regions. So far, the results seem decent. |
| [notebooks/cavity_flow/cavity_flow_summary.md](notebooks/cavity_flow/cavity_flow_summary.md) | April 28, 2026 | Further Cavity flow experiments with different boundary counditions | Attempted three boundary conditions - quartic polynomial, quadratic polynomial and smooth bump function. So far, only the quartic boundary yielded any good results. However, the smooth bump function boundary could prove useful later down the line. |