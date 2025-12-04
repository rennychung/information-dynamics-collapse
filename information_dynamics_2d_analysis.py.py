import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def get_2d_entropy(x, y, bins=50):
    # Calculate 2D Histogram
    H, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]], density=True)
    # Normalize to probabilities
    probs = H.flatten() / np.sum(H) + 1e-10
    # Return Shannon Entropy (normalized)
    return entropy(probs, base=2) / np.log2(bins*bins)

def get_grid_order_parameter(x, y):
    """
    Measures how well particles align with the potential valleys.
    Valleys are at cos(4pi*x) = 1.
    If random (Gas), average is ~0.
    If perfect crystal, average is ~1.
    """
    # Average of cos(4pi x) and cos(4pi y)
    order_x = np.cos(4 * np.pi * x)
    order_y = np.cos(4 * np.pi * y)
    return (np.mean(order_x) + np.mean(order_y)) / 2

def run_2d_analysis():
    # --- Parameters ---
    n_particles = 5000 
    n_steps = 50
    i_values = np.linspace(0, 1, 21)
    
    # Data Storage
    results = {
        'I': i_values,
        'entropy_structure': [],
        'order_structure': [],
        'entropy_control': []
    }
    
    # Snapshots for visualization (Low, Mid, High)
    snapshots = {}
    snap_indices = [0, 10, 20] # Indices corresponding to I=0.0, 0.5, 1.0
    
    print("Running 2D Analysis sweep (Structure vs Control)...")
    
    for idx, I in enumerate(i_values):
        # Physics Parameters
        temperature = (1 - I)**3 * 0.15
        force_strength = I * 0.15
        
        # --- 1. Structure Simulation ---
        x = np.random.uniform(0, 1, n_particles)
        y = np.random.uniform(0, 1, n_particles)
        
        for t in range(n_steps):
            fx = -np.sin(4 * np.pi * x)
            fy = -np.sin(4 * np.pi * y)
            nx = np.random.normal(0, 1, n_particles)
            ny = np.random.normal(0, 1, n_particles)
            
            x = x + (fx * force_strength) + (nx * np.sqrt(temperature))
            y = y + (fy * force_strength) + (ny * np.sqrt(temperature))
            x, y = x % 1.0, y % 1.0
            
        results['entropy_structure'].append(get_2d_entropy(x, y))
        results['order_structure'].append(get_grid_order_parameter(x, y))
        
        if idx in snap_indices:
            snapshots[I] = (x, y)
            
        # --- 2. Control Simulation (Flat Potential) ---
        x_c = np.random.uniform(0, 1, n_particles)
        y_c = np.random.uniform(0, 1, n_particles)
        
        for t in range(n_steps):
            # No Force!
            nx = np.random.normal(0, 1, n_particles)
            ny = np.random.normal(0, 1, n_particles)
            
            x_c = x_c + (nx * np.sqrt(temperature)) # Only noise
            y_c = y_c + (ny * np.sqrt(temperature))
            x_c, y_c = x_c % 1.0, y_c % 1.0
            
        results['entropy_control'].append(get_2d_entropy(x_c, y_c))

    # --- Plotting the Dashboard ---
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3)
    
    # 1. Scientific Evidence: Entropy Drop
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['I'], results['entropy_structure'], 'b-o', label='Information Force')
    ax1.plot(results['I'], results['entropy_control'], 'r--', label='Control (Noise Only)')
    ax1.set_title('2D Entropy (Thermodynamics)')
    ax1.set_xlabel('Information Flow (I)')
    ax1.set_ylabel('Shannon Entropy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scientific Evidence: Order Parameter
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['I'], results['order_structure'], 'g-o', label='Grid Coherence')
    ax2.set_title('Lattice Order (Crystallinity)')
    ax2.set_xlabel('I')
    ax2.set_ylabel('Order Parameter (0-1)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Visual Evidence: The Phase Transition
    # Plot the 3 snapshots in the bottom row
    snap_keys = sorted(snapshots.keys())
    for k, I in enumerate(snap_keys):
        ax = fig.add_subplot(gs[1, k])
        x, y = snapshots[I]
        ax.hist2d(x, y, bins=60, range=[[0, 1], [0, 1]], cmap='inferno')
        
        state_name = "Gas" if I < 0.1 else ("Crystal" if I > 0.9 else "Transition")
        ax.set_title(f'I = {I:.1f} ({state_name})')
        ax.axis('off')

    plt.suptitle('Information-Driven Symmetry Breaking in 2D', fontsize=16)
    plt.tight_layout()
    plt.savefig('information_dynamics_2d_dashboard.png')
    print("Analysis complete. Saved to 'information_dynamics_2d_dashboard.png'.")

if __name__ == "__main__":
    run_2d_analysis()
    input("Press Enter to exit...")
