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
| [notebooks/toy_problems/discontinuous_summary.md](notebooks/toy_problems/discontinuous_summary.md) | Not specified | Discontinuous function approximation benchmarks (1-D and 2-D) across heaviside-based and expert-characteristic models. | Indeed, it "prooved" MoE has the best potential to express piecewise smooth functions, once the domain decomposition is known. |