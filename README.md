# Overdamped Blue Moon Sampling with Second-Order Integrators

The Blue Moon sampling method uses constrained molecular simulations to confine a system to a hypersurface where a chosen reaction coordinate is held constant. By integrating the mean force along this coordinate, one can reconstruct the free energy surface.

This work presents the first application of the second-order accurate **Post-processor for Variable Diffusion (PVD-2)** integrator to Blue Moon sampling with overdamped Langevin dynamics (also known as Brownian or Constrained Overdamped Langevin Dynamics - COLD).

### Constrained Overdamped Langevin Dynamics (COLD)
To perform Blue Moon sampling, we need an ergodic sampler that respects the holonomic constraint. COLD is an ideal choice, described by the following stochastic differential equation (SDE) for a system with an isotropic mass matrix ($M=mI$):

$$dQ = \frac{1}{m\gamma}(-\Pi(Q) \nabla V(Q) + \beta^{-1} \operatorname{div} (\Pi(Q))dt + \sqrt{\frac{2}{m\gamma \beta}}\Pi(Q)dW$$

Where:
-   $Q$ represents the atomic coordinates.
-   $V(Q)$ is the potential energy.
-   $\gamma$ is the friction coefficient.
-   $\beta = (k_B T)^{-1}$.
-   $\Pi(Q)$ is a projection matrix that projects out motion along the reaction coordinate's gradient, ensuring the constraint is maintained.
-   $W$ is a standard Wiener process (Brownian motion).

## Numerical Integrators Studied

* **Euler-Maruyama (EM):** The standard, first-order weak integrator. It is simple but suffers from significant drift from the constraint.
* **PVD-2:** A second-order weak integrator for Brownian dynamics with variable diffusion. It shows improved constraint satisfaction over EM but still exhibits drift.
* **Projected Euler-Maruyama (p-EM):** A modified EM scheme that adds a "corrector" step. After each integration step, a root-finding algorithm (Newton-Raphson) solves for a small push that moves the system back onto the constraint manifold.
* **Projected PVD-2 (p-PVD-2):** The same projection/correction methodology applied to the PVD-2 integrator.
