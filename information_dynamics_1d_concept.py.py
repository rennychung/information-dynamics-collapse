import numpy as np
import matplotlib.pyplot as plt

def run_1d_bifurcation():
    n_particles = 10000
    n_steps = 60
    i_values = np.linspace(0, 1, 50) # High resolution for smooth heatmap
    
    heatmap_data = []

    print("Running 1D Bifurcation (The Fork in the Road)...")

    for I in i_values:
        x = np.random.uniform(0, 1, n_particles)
        
        # Physics: Cubic noise decay
        temperature = (1 - I)**3 * 0.15 
        force_strength = I * 0.15
        
        for t in range(n_steps):
            force = -np.sin(4 * np.pi * x) 
            noise = np.random.normal(0, 1, n_particles)
            x = x + (force * force_strength) + (noise * np.sqrt(temperature))
            x = x % 1.0 
            
        # Capture density for this slice of I
        counts, _ = np.histogram(x, bins=100, range=(0, 1), density=True)
        heatmap_data.append(counts)

    # Plot
    plt.figure(figsize=(10, 6))
    # Transpose so X=Information, Y=Position
    plt.imshow(np.array(heatmap_data).T, aspect='auto', origin='lower',
               extent=[0, 1, 0, 1], cmap='magma')
    
    plt.title('The Bifurcation: Emergence of Choice', fontsize=16)
    plt.xlabel('Information Flow (I)')
    plt.ylabel('Position (x)')
    plt.colorbar(label='Probability Density')
    
    plt.tight_layout()
    plt.savefig('information_dynamics_1d_concept.png')
    print("Saved 1D Bifurcation plot to 'information_dynamics_1d_concept.png'.")

if __name__ == "__main__":
    run_1d_bifurcation()
    # FIX: Explicitly show the plot
    plt.show() 
    input("Press Enter to exit...")
