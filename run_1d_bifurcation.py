import numpy as np
import matplotlib.pyplot as plt

# --- System Parameters ---
# T0 is the maximum effective temperature (when lambda = 0)
T0 = 0.15 
# Force scaling factor at lambda = 1
FORCE_SCALE = 0.15 
N_STEPS = 60
N_PARTICLES = 10000

def run_1d_bifurcation():
    """
    Simulates the 1D phase transition using Langevin dynamics, showing the 
    bifurcation as the control parameter lambda (λ) increases.
    """
    
    # Lambda values act as the single control parameter (0 to 1)
    lambda_values = np.linspace(0, 1, 50) 
    
    heatmap_data = []

    print("Running 1D Bifurcation (Symmetry Breaking)...")

    for lmbda in lambda_values:
        # Initialize particles uniformly across the domain [0, 1]
        x = np.random.uniform(0, 1, N_PARTICLES)
        
        # --- Coordinated Scaling Mechanism ---
        # 1. Effective Temperature (Noise Decay): T(λ) = T0 * (1 - λ)^3
        temperature = T0 * (1 - lmbda)**3
        
        # 2. Potential Scaling (Drift Term Strength): Scales with λ
        potential_strength = lmbda * FORCE_SCALE
        
        # Euler-Maruyama integration steps
        for t in range(N_STEPS):
            # Deterministic Force: -dV/dx for potential V(x) = sin(4*pi*x) / (4*pi)
            # The potential V(x) = (1/16pi^2) * cos(4*pi*x) provides 4 stable wells in [0, 1]
            force = -np.sin(4 * np.pi * x)
            
            # Stochastic Noise: standard normal distribution (dW_t)
            noise = np.random.normal(0, 1, N_PARTICLES)
            
            # Euler-Maruyama: x(t+dt) = x(t) + F(x, λ)dt + sqrt(2D(λ))dWt
            # (Note: dt=1 for simplicity; D = T/2, so sqrt(2*D) = sqrt(T))
            x = x + (force * potential_strength) + (noise * np.sqrt(temperature))
            
            # Enforce periodic boundary conditions
            x = x % 1.0
            
        # Capture density distribution for this slice of lambda
        counts, _ = np.histogram(x, bins=100, range=(0, 1), density=True)
        heatmap_data.append(counts)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Transpose so X=Control Parameter (lambda), Y=Position
    plt.imshow(np.array(heatmap_data).T, aspect='auto', origin='lower',
               extent=[0, 1, 0, 1], cmap='magma')
    
    plt.title('1D Bifurcation: Coordinated Cooling and Potential Rescaling', fontsize=16)
    
    # Use LaTeX for lambda (λ) for academic presentation
    plt.xlabel(r'Control Parameter $\lambda$', fontsize=14)
    plt.ylabel('Particle Position (x)', fontsize=14)
    plt.colorbar(label='Probability Density')
    
    # Update filename to reflect the new theme
    plt.tight_layout()
    plt.savefig('classical_phase_transition_1d_bifurcation.png')
    print("Saved 1D Bifurcation plot to 'classical_phase_transition_1d_bifurcation.png'.")

if __name__ == "__main__":
    # Ensure the plot runs successfully after the calculation
    run_1d_bifurcation()
    plt.show()
