import numpy as np
import matplotlib.pyplot as plt
import sys

# Wrap the import in try/except to catch missing scipy immediately
try:
    from scipy.stats import entropy
except ImportError:
    print("\nCRITICAL ERROR: 'scipy' library is missing.")
    print("Please install it using: pip install scipy")
    input("Press Enter to exit...")
    sys.exit(1)

# --- Helper Function: Phase Coherence (Order Parameter) ---
def phase_coherence(x_current, x_next):
    """
    Measures the average phase difference between particles from one step to the next.
    In the ordered state (crystal), this value is high.
    """
    dx = x_next - x_current
    return np.mean(np.cos(2 * np.pi * dx))

def run_hysteresis_simulation():
    print("Initializing Simulation...")
    
    # --- Simulation Parameters ---
    n_particles = 5000        
    steps_per_phase = 50      # Increased back to 50 for a smoother curve
    
    # *** PHYSICS TUNING FIX ***
    # Increasing the measurement time allows the coherence metric to register the lag better.
    relaxation_time = 3       
    
    # *** MOBILITY TUNING (KEEPING AT 0.03) ***
    mobility_factor = 0.03    
    
    # We define the path: Ramp Up (Freezing) followed by Ramp Down (Melting)
    i_ramp_up = np.linspace(0, 1, steps_per_phase)
    i_ramp_down = np.linspace(1, 0, steps_per_phase)
    
    # Storage for results
    history_up = {'I': [], 'H': [], 'C': []} 
    history_down = {'I': [], 'H': [], 'C': []}
    
    # Initialize particles
    x = np.random.uniform(0, 1, n_particles)
    
    print("Running Hysteresis Loop (Memory Test)...")
    
    # --- PHASE 1: FREEZING ---
    print("Phase 1: Information Ramp Up (Freezing)...")
    for I in i_ramp_up:
        # Tuned coefficients to 0.03 to ensure sufficient lag
        temperature = (1 - I)**3 * mobility_factor 
        force_strength = I * mobility_factor
        
        for _ in range(relaxation_time):
            force = -np.sin(4 * np.pi * x) 
            noise = np.random.normal(0, 1, n_particles)
            x_new = x + (force * force_strength) + (noise * np.sqrt(temperature))
            x_new = x_new % 1.0 
            
            # Measurements are taken on the last sub-step
            if _ == relaxation_time - 1:
                counts, _ = np.histogram(x_new, bins=60, range=(0, 1), density=True)
                probs = counts / np.sum(counts) + 1e-10 
                measure_h = entropy(probs, base=2) / np.log2(60)
                measure_c = phase_coherence(x, x_new)
            
            x = x_new 
            
        history_up['I'].append(I)
        history_up['H'].append(measure_h)
        history_up['C'].append(measure_c)

    # --- PHASE 2: MELTING ---
    print("Phase 2: Information Ramp Down (Melting)...")
    for I in i_ramp_down:
        # Tuned coefficients to 0.03 to ensure sufficient lag
        temperature = (1 - I)**3 * mobility_factor 
        force_strength = I * mobility_factor
        
        for _ in range(relaxation_time):
            force = -np.sin(4 * np.pi * x) 
            noise = np.random.normal(0, 1, n_particles)
            x_new = x + (force * force_strength) + (noise * np.sqrt(temperature))
            x_new = x_new % 1.0 
            
            # Measurements are taken on the last sub-step
            if _ == relaxation_time - 1:
                counts, _ = np.histogram(x_new, bins=60, range=(0, 1), density=True)
                probs = counts / np.sum(counts) + 1e-10 
                measure_h = entropy(probs, base=2) / np.log2(60)
                measure_c = phase_coherence(x, x_new)
            
            x = x_new
            
        history_down['I'].append(I)
        history_down['H'].append(measure_h)
        history_down['C'].append(measure_c)

    print("Simulation finished. Generating plots...")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Entropy Hysteresis
    plt.subplot(1, 2, 1)
    plt.plot(history_up['I'], history_up['H'], color='#3b82f6', label='Ramp Up (Freezing)', linewidth=2, alpha=0.8)
    plt.plot(history_down['I'], history_down['H'], color='#ef4444', label='Ramp Down (Melting)', linewidth=2, linestyle='--')
    plt.fill_between(history_up['I'], history_up['H'], history_down['H'], color='#9ca3af', alpha=0.15, label='Hysteresis Area')
    
    plt.title('Normalized Entropy Hysteresis (System Memory)', fontsize=14, fontweight='bold')
    plt.xlabel('Information Flow (I)', fontsize=12)
    plt.ylabel('Normalized Entropy (H)', fontsize=12)
    plt.legend(frameon=True, shadow=True, fancybox=True)
    plt.ylim(0, 1.05)
    
    # Plot 2: Coherence Hysteresis
    plt.subplot(1, 2, 2)
    plt.plot(history_up['I'], history_up['C'], color='#3b82f6', label='Ramp Up (Order Increase)', linewidth=2, alpha=0.8)
    plt.plot(history_down['I'], history_down['C'], color='#ef4444', label='Ramp Down (Order Decrease)', linewidth=2, linestyle='--')
    plt.fill_between(history_up['I'], history_up['C'], history_down['C'], color='#9ca3af', alpha=0.15)
    
    plt.title('Phase Coherence Hysteresis', fontsize=14, fontweight='bold')
    plt.xlabel('Information Flow (I)', fontsize=12)
    plt.ylabel('Phase Coherence', fontsize=12)
    plt.legend(frameon=True, shadow=True, fancybox=True)

    plt.tight_layout(pad=3.0)
    
    # Save file
    try:
        plt.savefig('information_dynamics_hysteresis.png')
        print("Plot saved to 'information_dynamics_hysteresis.png'")
    except Exception as e:
        print(f"Warning: Could not save image file: {e}")

    # Show plot
    print("Opening plot window...")
    plt.show()

if __name__ == "__main__":
    try:
        run_hysteresis_simulation()
    except Exception as e:
        # This block catches ANY crash and keeps the window open
        print("\n" + "="*40)
        print("AN ERROR OCURRED:")
        print(e)
        print("="*40 + "\n")
    finally:
        # This ensures input() runs whether it succeeds or fails
        input("Program finished. Press Enter to exit...")
