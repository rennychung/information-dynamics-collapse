# Information Dynamics: A Classical Model of Wavefunction Collapse

A classical stochastic model demonstrating information-driven phase transitions and the mechanism of symmetry breaking.


## Overview

This repository contains a Python-based computational model that simulates a system where a control parameter, the information flow ($I$), drives a phase transition analogous to thermodynamic cooling.

By balancing a stabilizing potential force against a thermal-like noise, the model reproduces the dynamics of symmetry breaking and phase transitions, offering an intuitive classical parallel to phenomena such as quantum wavefunction collapse and measurement.


## Methodology

The system's dynamics are governed by the following stochastic differential equation (SDE), where the noise and force terms are coupled via the control parameter $I$:

$$
dx = \left(-I \cdot \nabla V(x)\right) dt + \sqrt{D(I)} \cdot dW_t
$$

* **$I$** is the information control parameter ($0 \le I \le 1$).
* **$-I \cdot \nabla V(x)$** is the ordering force, increasing linearly with $I$.
* **$D(I)$** is the Noise Intensity (Diffusion Coefficient), which is proportional to $(1-I)^3$.
* **$dW_t$** is the increment of the Wiener process (Gaussian white noise).

The non-linear scaling of noise intensity, $D(I) \propto (1-I)^3$, is essential for modeling the rapid onset of ordering characteristic of a first-order phase transition.


## Key Findings (Model Validation)

The simulation results confirm the model's fundamental physical properties:

| Feature | Description | Supporting Figure |
| :--- | :--- | :--- |
| **Thermodynamic Analogy** | Increasing $I$ drives the system from a high-entropy "Gas" state (disordered) to a low-entropy "Crystal" state (ordered), behaving identically to thermodynamic cooling. | **1, 2** |
| **First-Order Transition** | The system exhibits hysteresis (system memory), proving that the transition is a first-order phase transition where the path to order differs from the path to disorder. | **4** |
| **Robustness** | The transition threshold is consistent across variations in the noise decay exponent, confirming the model's stability against parameter tuning. | **5** |
| **Causality** | Control group tests with a flat potential confirm that the drop in entropy and emergence of structure are **causally linked** to the potential's interaction with $I$. | **5** |
| **Universality** | The core phase transition phenomenon is observed consistently across 1D, 2D, and 3D simulations, and is independent of the exact potential shape. | **1, 2, 3, 5** |


## Visual Results

### 1. 1D Bifurcation and Core Analysis 

This figure illustrates the pitchfork bifurcation, the mathematical mechanism driving the phase change. As the information flow ($I$) increases past a critical value, the single, uniform probability distribution (high-entropy state) splits into multiple stable, localized peaks (low-entropy states). This transition represents the spontaneous symmetry breaking in the system's position.

<img width="1000" height="600" alt="information_dynamics_1d_concept" src="https://github.com/user-attachments/assets/7ec46d0a-b5fa-4390-a87c-9abff49b17f8" />



### 2. 2D Entropy and Crystallinity Dashboard 

This dashboard provides both quantitative and visual evidence in two dimensions. The sharp drop in shannon entropy (the thermodynamic measure) aligns perfectly with the rise in the lattice order parameter (the structural measure), demonstrating the transition from a disordered gas to an ordered crystal.



<img width="1400" height="1000" alt="information_dynamics_2d_dashboard" src="https://github.com/user-attachments/assets/9a630df3-42a2-4b42-af4d-8441215e3c0c" />

### 3. 3D Ensemble 

A visual confirmation of the phase progression in a three-dimensional lattice: from a uniformly spread gas state ($I=0.0$) through the clustering transition state ($I=0.6$), ending in the highly localized crystal state ($I=1.0$).


<img width="1500" height="600" alt="information_dynamics_3d_ensemble" src="https://github.com/user-attachments/assets/917daad0-3cee-461a-a292-297f0a021eb2" />

### 4. Normalized Entropy Hysteresis 

This plot shows the separation between the ramp Up (Freezing) and ramp down (Melting) curves for Normalized Entropy. The resulting hysteresis area proves that the system's state is history-dependent, confirming the transition is a first-order phase transition.


<img width="1400" height="600" alt="information_dynamics_hysteresis" src="https://github.com/user-attachments/assets/841ff4a7-d141-4ee1-b433-ee261b2b9a97" />

### 5. Stress Test 

This multi-panel figure confirms the model's stability and scientific validity by showing robustness (varying noise exponents), universality (different potentials), and the strong causality provided by the control group test.


<img width="1800" height="600" alt="information_dynamics_stress_model" src="https://github.com/user-attachments/assets/cb4503b4-0eff-4d62-bd64-1fa78d57e808" />


## Repository Contents

* `information_dynamics_1d_concept.py`: Generates the 1D bifurcation plot.
* `information_dynamics_2d_analysis.py`: Runs the 2D sweep and generates the dashboard plot.
* `information_dynamics_3d_stats.py`: Runs the 3D ensemble simulation and visualization.
* `information_dynamics_hysteresis.py`: Runs the freezing/melting cycle to generate the hysteresis loop.
* `information_dynamics_stress_test.py`: Runs the robustness, universality, and causality validation tests.
* `Paper_Draft.md`: Preliminary manuscript on methodology and theoretical implications.


## Usage

### Prerequisites

* `numpy`
* `matplotlib`
* `scipy`

### Running the Simulation

To run the full validation suite and generate the stress test figure:
information_dynamics_stress_test.py

**Citation**
If you use this model in your research or teaching, please cite:

Renny Chung, "Information-Driven Phase Transitions in Stochastic Systems," 2025
