# Discontinuous Function Approximation — Results Summary

Three neural network architectures are compared on their ability to approximate
discontinuous functions in 1-D and 2-D.

---

## Architectures

| Name | Description |
|---|---|
| **tanh + heavyside** | Standard MLP with `tanh` activations and a Heaviside output activation to induce a hard jump. |
| **half-heaviside** | MLP with a smooth half-Heaviside last-layer activation; differentiable approximation of the jump. |
| **experts with χ** | Partition-of-unity expert network: two sub-MLPs weighted by a characteristic function χ, each responsible for one side of the discontinuity. |

All architectures share the same hidden layer widths `[32, 32, 64]` for the MLP baselines
and `[16, 16]` per expert for the per-characteristic model. Training: Adam, lr=1e-4, 20 000 epochs.

---

## Benchmark functions

| Function | Domain | Discontinuity |
|---|---|---|
| **g** | $[0, 2\pi]$ | Single jump at $x = \pi/2$; piecewise $\{x,\; -\sin x - 2\}$ |
| **g1** | $[0, 2\pi]$ | Four regions with distinct smooth branches |
| **g2** | $[0,2\pi]^2$ | Curved interface $y = \pi + \sin x$ separating two smooth 2-D fields |

---

## L2 errors

### g — 1-D single-jump

| Model | L2 error |
|---|---|
| tanh + heavyside | `0.262790` |
| half-heaviside   | `0.147337` |
| experts χ        | `0.031496` |

### g1 — 1-D four-region

| Model | L2 error |
|---|---|
| tanh + heavyside | `0.347215` |
| half-heaviside   | `0.351180` |
| experts χ        | `0.276456` |

### g2 — 2-D curved interface

| Model | L2 error |
|---|---|
| tanh + heavyside | `0.335894` |
| half-heaviside   | `0.334211` |
| experts χ        | `0.103831` |

---

## Observations

- **tanh + heavyside** and **half-heaviside**: there appeared to be a certain level of disapointement from the experiment result. As the Heaviside function was present mainly in the last and next to last layer, the network focused mostly on forcing the heaviside jump size to zero. As Heaviside function has zero gradient almost everywhere, the network is unable to optimize "center of jump". An STE strategy was employed in order to combat this shortcoming, and indeed it resulted in the correct parameters changing, however, it produced barely negligible results. Further experimentation may be required to arrive at a decisive conclusion about the effectiveness of using Heaviside function.

- **experts χ**: as expected, knowing the characteristic function of the regions of smoothness and using them to split the network yields the cleanest possible results, as each (disconected) part of the model could focus on learning its specific region, leaving determination of discontinuity to the user. This could be especially usefull when, as mentioned above, the characteristic functions are known apriori, which is (hopefuly) the case most of the time. Also note, that by virtue of specialization to regions, fewer parameters may be used to achieve same or even better results when compared to standard MLP (not tested) or MLP enriched by Heaviside (so far).

- When it came to the difficulty of learning, it seems that in this category, all three methods performed about the same.

## Further work

- It is still unclear, wheter there exist better Heaviside architecture than those tested.
- Over time, more functions and benchmarks may be added.
