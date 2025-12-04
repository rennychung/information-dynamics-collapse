import numpy as np
import matplotlib.pyplot as plt
import sys

# Wrap the import in try/except to catch missing scipy immediately
try:
    from scipy.stats import entropy
except ImportError:
    print("\nCRITICAL ERROR: 'scipy' library is missing.")
    print("Please install it using: pip install scipy")
    # This ensures exit if scipy is missing, allowing the final input() to be the only one.
    sys.exit(1)


def run_variant(n_particles=2000, exponent=3, potential_type='cosine'):
    """
    Runs a single simulation run under specific parameters (exponent or potential shape).
    The steps are set high (200) to ensure the system reaches equilibrium quickly.
    """
    # Using a higher number of steps to ensure equilibrium for these tests
    n_steps = 200
    # Increased to 50 points for smoother lines
    lambda_values = np.linspace(0, 1, 50) # Renamed from i_values 
    avg_entropies = []
    
    # CRITICAL: I replaced with lambda_param
    for lambda_param in lambda_values:
        # Initialize particles randomly at the start of each lambda_param value sweep
        x = np.random.uniform(0, 1, n_particles)
        
        # Physics Parameters (using original 0.15 since this is NOT the hysteresis test)
        # As lambda_param goes 0 to 1, temperature decreases and force strength increases
        temperature = (1 - lambda_param)**exponent * 0.15 # I replaced with lambda_param
        force_strength = lambda_param * 0.15 # I replaced with lambda_param
        
        for t in range(n_steps):
            # 1. Define Potential / Force
            if potential_type == 'cosine':
                # Original potential: Multi-well (2 periods in length 1)
                force = -np.sin(4 * np.pi * x)
            elif potential_type == 'double_well':
                # Classic Quartic Double Well: V = x^4 - x^2 centered at 0.5
                # Shift x to -1 to 1 range for calc, then force back
                x_shifted = (x - 0.5) * 2
                # Force is -dV/dx = -(4*x_shifted**3 - 2*x_shifted)
                f_shifted = -(4 * x_shifted**3 - 2 * x_shifted) 
                force = f_shifted
            elif potential_type == 'flat':
                # Control Group: No structure at all
                force = np.zeros_like(x)
                
            noise = np.random.normal(0, 1, n_particles)
            
            # Update (Langevin Dynamics)
            x_new = x + (force * force_strength) + (noise * np.sqrt(temperature))
            x_new = x_new % 1.0 # Periodic Boundary Conditions (BC)
            
            # Update position for next step's force calculation
            x = x_new
            
            # Measure (only on the last step after full relaxation)
            if t == n_steps - 1:
                # Calculate Entropy (S)
                counts, _ = np.histogram(x, bins=60, range=(0, 1), density=True)
                # Add tiny amount to avoid log(0)
                probs = counts / np.sum(counts) + 1e-10 
                # Normalize by log2(bins) so S_max = 1
                e = entropy(probs, base=2) / np.log2(60)
                avg_entropies.append(e)
            
            
    return lambda_values, avg_entropies # Renamed return value

def run_stress_test():
    """Executes the three validation tests and generates a combined plot."""
    
    # Renamed description
    print("Starting Phase Transition Parameter Test Suite...")
    
    # Using a cleaner style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(18, 6)) # Wider figure for three plots
    
    # --- TEST 1: The "Robustness" Test (Noise Sensitivity) ---
    print("Running Test 1: Noise Exponents (Robustness)...")
    plt.subplot(1, 3, 1)

    # Use a list of styles to ensure separation
    styles = ['o-', 's-', '^-'] 
    
    # Testing how sensitive the transition is to the rate the noise decays
    for i, exp in enumerate([2, 3, 4]):
        # Renamed variable
        lambda_val, ent = run_variant(exponent=exp)
        # FIX APPLIED HERE: Used raw f-string (rf'...') to resolve SyntaxWarning
        plt.plot(lambda_val, ent, styles[i], linewidth=2, markersize=3, label=rf'Noise Decay Power (1-$\lambda$)^{exp}')
    
    plt.title('Robustness: Effect of Cooling Rate Exponent', fontsize=14)
    # Fixed SyntaxWarning using raw string (r'...') and updated label
    plt.xlabel(r'Control Parameter $\lambda$', fontsize=12) 
    plt.ylabel('Normalized Entropy', fontsize=12)
    # Legend fixed to 'lower left' to avoid collision.
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    # --- TEST 2: The "Universality" Test (Potential Shapes) ---
    print("Running Test 2: Potential Shapes (Universality)...")
    plt.subplot(1, 3, 2)
    shapes = ['cosine', 'double_well']
    
    # Testing if the phase transition still occurs with a different underlying structure
    for i, shape in enumerate(shapes):
        # Renamed variable
        lambda_val, ent = run_variant(potential_type=shape)
        plt.plot(lambda_val, ent, styles[i], linewidth=2, markersize=3, label=f'Potential: {shape}')
        
    plt.title('Universality: Potential Shape Dependence', fontsize=14)
    # Fixed SyntaxWarning using raw string (r'...') and updated label
    plt.xlabel(r'Control Parameter $\lambda$', fontsize=12) 
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)

    # --- TEST 3: The "Negative Control" Test (Causality) ---
    print("Running Test 3: The Control Group (Causality)...")
    plt.subplot(1, 3, 3)
    
    # Run standard (Experimental Group)
    lambda_exp, ent_exp = run_variant(potential_type='cosine') # Renamed variable
    plt.plot(lambda_exp, ent_exp, 'b-o', linewidth=2, markersize=3, label='Experimental (Cosine Structure)')
    
    # Run flat (Control Group)
    lambda_ctrl, ent_ctrl = run_variant(potential_type='flat') # Renamed variable
    plt.plot(lambda_ctrl, ent_ctrl, 'r--', linewidth=2, label='Control (Flat Structure)')
    
    plt.title('Causality: Structure vs. No Structure', fontsize=14)
    # Fixed SyntaxWarning using raw string (r'...') and updated label
    plt.xlabel(r'Control Parameter $\lambda$', fontsize=12) 
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    # Renamed output file
    plt.savefig('phase_transition_parameter_test_results.png')
    print("Parameter test suite complete. Saved to 'phase_transition_parameter_test_results.png'.")
    plt.show()

if __name__ == "__main__":
    try:
        run_stress_test()
    except Exception as e:
        # Print error details if an exception occurs
        print("\n" + "="*40)
        print("AN ERROR OCCURRED:")
        print(e)
        print("="*40 + "\n")
    
    # This line is outside the try/except block and ensures the program waits,
    # and the prompt matches your exact request.
    input("Press Enter to exit...")
