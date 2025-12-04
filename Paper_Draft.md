# Coordinated Phase Transitions: A Minimal Classical Model

## Abstract

We present a minimal numerical model for classical first-order phase transitions using Langevin dynamics. By coordinating the scaling of potential strength and effective temperature via a single control parameter, we demonstrate robust symmetry breaking and hysteresis in 1D, 2D, and 3D systems.

## Introduction

Phase transitions and symmetry breaking are foundational in statistical physics. Our model investigates the minimal requirements for inducing first-order transitions in classical, non-equilibrium systems, using a stochastically driven potential landscape.

## Model Formulation

- **Equation of Motion (SDE):**

  dx = -λ ∇V(x) dt + D(λ) dW_t
  where:

  - **Potential:**  
    `V(x) = V0 * sum_i cos(4π x_i)`

  - **Diffusion coefficient:**  
    `D(λ) = D0 * (1 - λ)^3`

  - **λ:**  
    Control parameter, λ ∈ [0, 1]  
    dW_t is a standard Wiener process (Gaussian noise).

- **Initial/Boundary Conditions:**  
  Uniform initial distribution, periodic boundaries.

- **Parameter Table:**

  | Parameter           | Description              | Example Value      |
  |---------------------|--------------------------|--------------------|
  | D                   | Dimension                | 1, 2, 3            |
  | N_particles         | Number of particles      | 1000               |
  | N_steps             | Integration steps        | 10000–100000       |
  | V0                  | Potential strength       | 1.0                |
  | D0                  | Diffusion scale          | 0.5                |
  | γ                   | Cooling exponent         | 3                  |
  | λ                   | Control parameter        | [0, 1] sweep       |
  | Δt                  | Time step                | 0.001              |

## Methods

- Numerical integration (Euler-Maruyama)
- Parameter sweeps over λ
- Ensemble statistics (multiple runs)
- Visualization (matplotlib)

## Experiments

- **1D Bifurcation:** Visualizes density splitting as λ increases.
- **2D Dashboard:** Plots entropy and order parameter, demonstrates critical point.
- **3D Ensemble:** Shows bulk transition in particle distributions.
- **Hysteresis:** Confirms first-order nature via loop area.
- **Stress Tests:** Robustness to dimension, particle count, noise.

## Results

- Coordinated control is necessary for the transition; cooling alone is insufficient.
- Sharp drop in entropy and rise in order parameter at a critical value of λ.
- Hysteresis confirms first-order behavior.

## Discussion

- **Minimality:** Only one control parameter required for symmetry breaking.
- **Applicability:** Insights into other driven-dissipative and non-equilibrium systems.
- **Comparison:** Related to classical models (Ising, XY, Fokker–Planck, etc.).

## Limitations & Future Work

- Model simplicity (ignores long-range interactions, quantum effects).
- Future: Explore disorder, external driving, more complex potentials, and finite-size scaling.

## Reproducibility

All code and parameters are included in this repository.  
See script headers and [README.md](./README.md) for exact settings.

## References

1. Stanley, H.E. *Phase Transitions and Critical Phenomena*, Oxford, 1971.
2. Risken, H. *The Fokker–Planck Equation*, Springer, 1996.
3. [Add more as relevant]

