import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# --- System Parameters ---
T0 = 0.15 # Maximum effective temperature (at lambda = 0)
FORCE_SCALE = 0.15 # Potential scaling factor (at lambda = 1)
N_PARTICLES = 5000
N_STEPS = 50

def get_2d_entropy(x, y, bins=50):
    """
    Calculates the normalized Shannon entropy based on the 2D particle density.
    Normalized relative to the maximum possible entropy for the given bins.
    """
    # Calculate 2D Histogram
    H, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]], density=True)
    # Normalize to probabilities
    probs = H.flatten() / np.sum(H) + 1e-10
    # Return Shannon Entropy (normalized)
    return entropy(probs, base=2) / np.log2(bins*bins)

def get_grid_order_parameter(x, y):
    """
    Calculates the lattice order parameter, measuring how well particles align 
    with the 2D periodic potential valleys (cos(4pi*x) and cos(4pi*y)).
    """
    # Average of cos(4pi x) and cos(4pi y)
    order_x = np.cos(4 * np.pi * x)
    order_y = np.cos(4 * np.pi * y)
    return (np.mean(order_x) + np.mean(order_y)) / 2

def run_2d_analysis():
    # Lambda values for the sweep
    lambda_values = np.linspace(0, 1, 21)
    
    # Data Storage - Updated keys to use lambda
    results = {
        'lambda_param': lambda_values,
        'entropy_structure': [],
        'order_structure': [],
        'entropy_control': []
    }
    
    # Snapshots for visualization (Low, Mid, High)
    snapshots = {}
    snap_indices = [0, 10, 20] # Indices corresponding to lambda=0.0, 0.5, 1.0
    
    print("Running 2D Analysis sweep (Structure vs Control)...")
    
    for idx, lmbda in enumerate(lambda_values):
        # Physics Parameters (Coordinated Scaling)
        temperature = (1 - lmbda)**3 * T0
        potential_strength = lmbda * FORCE_SCALE
        
        # --- 1. Structure Simulation (with Potential) ---
        x = np.random.uniform(0, 1, N_PARTICLES)
        y = np.random.uniform(0, 1, N_PARTICLES)
        
        for t in range(N_STEPS):
            # Deterministic Force: F = -λ*grad(V)
            fx = -np.sin(4 * np.pi * x)
            fy = -np.sin(4 * np.pi * y)
            
            # Stochastic Noise
            nx = np.random.normal(0, 1, N_PARTICLES)
            ny = np.random.normal(0, 1, N_PARTICLES)
            
            # Update positions (Euler-Maruyama)
            x = x + (fx * potential_strength) + (nx * np.sqrt(temperature))
            y = y + (fy * potential_strength) + (ny * np.sqrt(temperature))
            x, y = x % 1.0, y % 1.0
            
        results['entropy_structure'].append(get_2d_entropy(x, y))
        results['order_structure'].append(get_grid_order_parameter(x, y))
        
        if idx in snap_indices:
            snapshots[lmbda] = (x, y)
            
        # --- 2. Control Simulation (Flat Potential, same cooling) ---
        x_c = np.random.uniform(0, 1, N_PARTICLES)
        y_c = np.random.uniform(0, 1, N_PARTICLES)
        
        for t in range(N_STEPS):
            # Only Noise Term (Force = 0)
            nx = np.random.normal(0, 1, N_PARTICLES)
            ny = np.random.normal(0, 1, N_PARTICLES)
            
            x_c = x_c + (nx * np.sqrt(temperature)) # Only noise
            y_c = y_c + (ny * np.sqrt(temperature))
            x_c, y_c = x_c % 1.0, y_c % 1.0
            
        results['entropy_control'].append(get_2d_entropy(x_c, y_c))

    # --- Plotting the Dashboard ---
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Use raw string for suptitle to include lambda
    plt.suptitle(r'Minimal Classical Model: 2D Phase Transition Dashboard ($\lambda$ Sweep)', fontsize=16)
    
    # 1. Scientific Evidence: Entropy Drop
    ax1 = fig.add_subplot(gs[0, 0])
    # Updated plot labels and variable names
    ax1.plot(results['lambda_param'], results['entropy_structure'], 'b-o', label='Coordinated Scaling (Potential + Cooling)')
    ax1.plot(results['lambda_param'], results['entropy_control'], 'r--', label='Control (Cooling Only)')
    ax1.set_title('Normalized Shannon Entropy Drop')
    ax1.set_xlabel(r'Control Parameter $\lambda$')
    ax1.set_ylabel('Normalized Entropy (H)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Scientific Evidence: Order Parameter
    ax2 = fig.add_subplot(gs[0, 1])
    # Updated plot labels and variable names
    ax2.plot(results['lambda_param'], results['order_structure'], 'g-o', label='Grid Coherence')
    ax2.set_title('Lattice Order Parameter (Crystallinity)')
    ax2.set_xlabel(r'Control Parameter $\lambda$')
    ax2.set_ylabel('Order Parameter (M)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Visual Evidence: The Phase Transition Snapshots
    # Plot the 3 snapshots in the bottom row
    snap_keys = sorted(snapshots.keys())
    for k, lmbda_val in enumerate(snap_keys):
        ax = fig.add_subplot(gs[1, k])
        x, y = snapshots[lmbda_val]
        ax.hist2d(x, y, bins=60, range=[[0, 1], [0, 1]], cmap='inferno')
        
        state_name = "Disordered Gas" if lmbda_val < 0.1 else ("Ordered Crystal" if lmbda_val > 0.9 else "Transition Zone")
        
        # Use raw string for title to include lambda
        ax.set_title(rf'$\lambda$ = {lmbda_val:.1f} ({state_name})')
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    # Save the plot with the new filename
    plt.savefig('classical_phase_transition_2d_dashboard.png')
    print("Analysis complete. Saved to 'classical_phase_transition_2d_dashboard.png'.")
    
    # Display the plot window
    plt.show()

if __name__ == "__main__":
    run_2d_analysis()
    # This line holds the script open until the user presses Enter
    input("Press Enter to exit...")
