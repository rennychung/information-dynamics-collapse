import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D 

# --- System Parameters ---
T0 = 0.1 # Maximum effective temperature (at lambda = 0)
FORCE_SCALE = 0.1 # Potential scaling factor (at lambda = 1)

def get_3d_entropy(x, y, z, bins=20):
    """
    Calculates the normalized Shannon entropy (relative to maximum entropy) 
    based on the particle density distribution.
    """
    H, _ = np.histogramdd((x, y, z), bins=(bins, bins, bins), range=[(0,1), (0,1), (0,1)], density=True)
    # Flatten and normalize to sum to 1. Add small epsilon (1e-10) to avoid log(0).
    probs = H.flatten() / np.sum(H) + 1e-10
    max_entropy = np.log2(bins**3)
    return entropy(probs, base=2) / max_entropy

def get_3d_order_parameter(x, y, z):
    """
    Calculates a lattice-based order parameter, M = <cos(4*pi*x) + cos(4*pi*y) + cos(4*pi*z)>/3.
    This value approaches 1 for perfect ordering (particles in the well centers) 
    and 0 for a disordered gas.
    """
    cos_x = np.cos(4 * np.pi * x)
    cos_y = np.cos(4 * np.pi * y)
    cos_z = np.cos(4 * np.pi * z)
    return (np.mean(cos_x) + np.mean(cos_y) + np.mean(cos_z)) / 3

def run_simulation_trial(lmbda, n_particles, n_steps):
    """
    Runs a single 3D simulation trial using the control parameter lambda (λ).
    Returns the final entropy, order parameter, and particle positions.
    """
    x = np.random.uniform(0, 1, n_particles)
    y = np.random.uniform(0, 1, n_particles)
    z = np.random.uniform(0, 1, n_particles)
    
    # --- Coordinated Scaling Mechanism ---
    # 1. Effective Temperature (Noise Decay): T(λ) = T0 * (1 - λ)^3
    temperature = T0 * (1 - lmbda)**3
    
    # 2. Potential Scaling (Drift Term Strength): Scales with λ
    potential_strength = lmbda * FORCE_SCALE
    
    # Euler-Maruyama integration loop
    for t in range(n_steps):
        # Deterministic Force components: F = -λ*grad(V)
        fx = -np.sin(4 * np.pi * x)
        fy = -np.sin(4 * np.pi * y)
        fz = -np.sin(4 * np.pi * z)
        
        # Stochastic Noise
        nx = np.random.normal(0, 1, n_particles)
        ny = np.random.normal(0, 1, n_particles)
        nz = np.random.normal(0, 1, n_particles)
        
        # Update positions
        x = x + (fx * potential_strength) + (nx * np.sqrt(temperature))
        y = y + (fy * potential_strength) + (ny * np.sqrt(temperature))
        z = z + (fz * potential_strength) + (nz * np.sqrt(temperature))
        
        # Enforce periodic boundary conditions
        x, y, z = x % 1.0, y % 1.0, z % 1.0
        
    H = get_3d_entropy(x, y, z)
    S = get_3d_order_parameter(x, y, z)
    return H, S, x, y, z

def run_ensemble_analysis():
    # --- Parameters ---
    N_PARTICLES = 5000 
    N_STEPS = 50
    N_TRIALS = 5 # Number of times to repeat the experiment
    
    # Lambda values for the visual snapshots
    lambda_snapshots = [0.0, 0.6, 1.0]
    
    # Store data for the final plot (from the last trial)
    final_viz_data = []

    print(f"Running Ensemble Analysis ({N_TRIALS} trials per λ-value)...\n")
    # Updated column headers to use lambda
    print(f"{'Lambda (λ)':<10} | {'Entropy (Mean ± Std)':<25} | {'Order (Mean ± Std)':<25}")
    print("-" * 65)
    
    for lmbda in lambda_snapshots:
        trial_entropies = []
        trial_orders = []
        
        # Run multiple trials
        for _ in range(N_TRIALS):
            H, S, x, y, z = run_simulation_trial(lmbda, N_PARTICLES, N_STEPS)
            trial_entropies.append(H)
            trial_orders.append(S)
        
        # Calculate Statistics
        avg_H = np.mean(trial_entropies)
        std_H = np.std(trial_entropies)
        
        avg_S = np.mean(trial_orders)
        std_S = np.std(trial_orders)
        
        # Updated print statement to use lambda
        print(f"{lmbda:<10.2f} | {avg_H:.4f} ± {std_H:.4f}             | {avg_S:.4f} ± {std_S:.4f}")
        
        # Save the last trial's positions for the 3D plot
        final_viz_data.append((lmbda, x, y, z, avg_S))

    # --- Plotting (Visuals of one representative trial) ---
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('3D Ensemble Snapshots: Disordered Gas to Ordered Crystal Phase Transition', fontsize=16)

    for idx, (lmbda_val, x, y, z, S_val) in enumerate(final_viz_data):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        ax.scatter(x, y, z, c=z, cmap='viridis', s=2, alpha=0.6)
        
        state_name = "Disordered Gas" if lmbda_val < 0.1 else ("Ordered Crystal" if lmbda_val > 0.9 else "Transition Zone")
        
        # FIX: Changed f-string to a raw f-string (rf) to handle the \lambda escape sequence correctly
        title_str = rf"{state_name} ($\lambda$={lmbda_val:.1f})\nOrder: {S_val:.2f}"
        ax.set_title(title_str, fontsize=11)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Clean up panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Adjusted top margin for the suptitle
    
    # Updated filename to reflect the new theme
    plt.savefig('classical_phase_transition_3d_ensemble.png')
    print("\nEnsemble analysis complete. Plot saved to 'classical_phase_transition_3d_ensemble.png'.")

if __name__ == "__main__":
    run_ensemble_analysis()
    plt.show()
