import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D # Explicit import for 3D

def get_3d_entropy(x, y, z, bins=20):
    H, _ = np.histogramdd((x, y, z), bins=(bins, bins, bins), range=[(0,1), (0,1), (0,1)], density=True)
    probs = H.flatten() / np.sum(H) + 1e-10
    max_entropy = np.log2(bins**3)
    return entropy(probs, base=2) / max_entropy

def get_3d_order_parameter(x, y, z):
    cos_x = np.cos(4 * np.pi * x)
    cos_y = np.cos(4 * np.pi * y)
    cos_z = np.cos(4 * np.pi * z)
    return (np.mean(cos_x) + np.mean(cos_y) + np.mean(cos_z)) / 3

def run_simulation_trial(I, n_particles, n_steps):
    """Runs a single simulation trial and returns stats + final positions."""
    x = np.random.uniform(0, 1, n_particles)
    y = np.random.uniform(0, 1, n_particles)
    z = np.random.uniform(0, 1, n_particles)
    
    temperature = (1 - I)**3 * 0.1
    force_strength = I * 0.1
    
    for t in range(n_steps):
        fx = -np.sin(4 * np.pi * x)
        fy = -np.sin(4 * np.pi * y)
        fz = -np.sin(4 * np.pi * z)
        
        nx = np.random.normal(0, 1, n_particles)
        ny = np.random.normal(0, 1, n_particles)
        nz = np.random.normal(0, 1, n_particles)
        
        x = x + (fx * force_strength) + (nx * np.sqrt(temperature))
        y = y + (fy * force_strength) + (ny * np.sqrt(temperature))
        z = z + (fz * force_strength) + (nz * np.sqrt(temperature))
        
        x, y, z = x % 1.0, y % 1.0, z % 1.0
        
    H = get_3d_entropy(x, y, z)
    S = get_3d_order_parameter(x, y, z)
    return H, S, x, y, z

def run_ensemble_analysis():
    # --- Parameters ---
    n_particles = 5000 
    n_steps = 50
    n_trials = 5  # Number of times to repeat the experiment
    i_snapshots = [0.0, 0.6, 1.0]
    
    # Store data for the final plot (from the last trial)
    final_viz_data = []

    print(f"Running Ensemble Analysis ({n_trials} trials per I-value)...\n")
    print(f"{'I':<10} | {'Entropy (Mean ± Std)':<25} | {'Order (Mean ± Std)':<25}")
    print("-" * 65)
    
    for I in i_snapshots:
        trial_entropies = []
        trial_orders = []
        
        # Run multiple trials
        for _ in range(n_trials):
            H, S, x, y, z = run_simulation_trial(I, n_particles, n_steps)
            trial_entropies.append(H)
            trial_orders.append(S)
        
        # Calculate Statistics
        avg_H = np.mean(trial_entropies)
        std_H = np.std(trial_entropies)
        
        avg_S = np.mean(trial_orders)
        std_S = np.std(trial_orders)
        
        print(f"{I:<10.2f} | {avg_H:.4f} ± {std_H:.4f}          | {avg_S:.4f} ± {std_S:.4f}")
        
        # Save the last trial's positions for the 3D plot
        final_viz_data.append((I, x, y, z, avg_S))

    # --- Plotting (Visuals of one representative trial) ---
    fig = plt.figure(figsize=(15, 6)) # Made figure slightly taller
    
    for idx, (I_val, x, y, z, S_val) in enumerate(final_viz_data):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        ax.scatter(x, y, z, c=z, cmap='viridis', s=2, alpha=0.6)
        
        state_name = "Gas" if I_val < 0.1 else ("Crystal" if I_val > 0.9 else "Transition")
        
        # Use simple string formatting for title to avoid encoding issues
        title_str = f"{state_name} (I={I_val:.1f})\nOrder: {S_val:.2f}"
        ax.set_title(title_str, fontsize=11)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Clean up panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # FIX: Adjust layout so titles aren't cut off
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 
    
    # FIX: Change extension to .jpg to match README.md
    plt.savefig('information_dynamics_3d_ensemble.jpg') 
    print("\nEnsemble analysis complete. Plot saved.")

if __name__ == "__main__":
    run_ensemble_analysis()
    plt.show() # Explicitly call show for robustness
    input("Press Enter to exit...")
