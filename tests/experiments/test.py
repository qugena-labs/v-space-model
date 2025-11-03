import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import expm

def create_potential_landscape(num_sites, steps):
    """Creates a potential energy landscape with steps."""
    potential = np.zeros(num_sites)
    for position, height in steps:
        potential[position:] = height
    return potential

def create_hamiltonian(num_sites, potential, hopping_strength=1.0):
    """Creates the Hamiltonian for a 1D tight-binding model."""
    hamiltonian = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        hamiltonian[i, i] = potential[i] + 2 * hopping_strength
        if i > 0:
            hamiltonian[i, i - 1] = -hopping_strength
        if i < num_sites - 1:
            hamiltonian[i, i + 1] = -hopping_strength
    return hamiltonian

def create_initial_state(num_sites, position, width, momentum=0.0):
    """Creates a Gaussian wave packet as the initial state (purely classical)."""
    x = np.arange(num_sites)
    gaussian = np.exp(-(x - position)**2 / (2 * width**2))
    momentum_phase = np.exp(1j * momentum * x)
    wave_packet = gaussian * momentum_phase
    wave_packet /= np.linalg.norm(wave_packet)  # Normalize
    return wave_packet

def run_simulation(hamiltonian, initial_state, time_steps, dt):
    """Simulates the time evolution of the system using classical matrix operations."""
    states_over_time = []
    current_state = initial_state.copy()
    
    # Time evolution operator
    U = expm(-1j * hamiltonian * dt)
    
    for _ in range(time_steps):
        current_state = U @ current_state
        states_over_time.append(current_state.copy())
    
    return states_over_time, initial_state

def animate_simulation(states_over_time, potential, dt, hamiltonian, initial_vector):
    """Creates an animation of the wave packet evolution."""
    fig, ax = plt.subplots()
    num_sites = len(potential)
    x = np.arange(num_sites)
    
    # Calculate total energy
    total_energy = np.real(np.vdot(initial_vector, hamiltonian @ initial_vector))
    
    # Determine plot bounds
    min_y = min(0, np.min(potential))
    max_y = max(total_energy, np.max(potential))
    plot_range = max_y - min_y if max_y > min_y else 1.0
    
    ax.set_ylim(min_y - 0.1 * plot_range, max_y + 0.2 * plot_range)

    # Plot energy and potential
    ax.axhline(y=total_energy, color='r', linestyle='--', label=f"Total Energy = {total_energy:.2f}")
    ax.plot(x, potential, label="Potential Energy", color='gray')
    
    # Scale the probability density for visualization
    max_initial_prob = np.max(np.abs(states_over_time[0])**2)
    visual_peak_height = 0.2
    prob_scaling_factor = visual_peak_height / max_initial_prob if max_initial_prob > 0 else 1

    line, = ax.plot(x, prob_scaling_factor * np.abs(states_over_time[0])**2, label="Wave Packet Probability")
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.set_xlabel("Position")
    ax.set_ylabel("Energy")
    ax.set_title("Quantum Wave Packet Evolution (Classical Simulation)")
    ax.legend()

    def update(frame):
        line.set_ydata(prob_scaling_factor * np.abs(states_over_time[frame])**2)
        time_text.set_text(f"Time = {frame * dt:.2f}")
        return line, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(states_over_time), blit=True, interval=50)
    plt.show()

if __name__ == '__main__':
    # --- Simulation Parameters ---
    NUM_SITES = 256  # Just use a regular array size
    
    # Define potential steps (position, height)
    POTENTIAL_STEPS = [(100, 0.15), (180, 0.3)]
    
    # Initial wave packet parameters
    INITIAL_POSITION = 50
    INITIAL_WIDTH = 10
    INITIAL_MOMENTUM = 0.45  # This gives the wave packet a "kick"
    
    # Time evolution parameters
    TIME_STEPS = 600
    DT = 0.5  # Time step size

    # --- Setup and Run ---
    potential = create_potential_landscape(NUM_SITES, POTENTIAL_STEPS)
    hamiltonian = create_hamiltonian(NUM_SITES, potential)
    initial_state = create_initial_state(NUM_SITES, INITIAL_POSITION, INITIAL_WIDTH, INITIAL_MOMENTUM)
    
    print("Running simulation...")
    states, initial_vector = run_simulation(hamiltonian, initial_state, TIME_STEPS, DT)
    print("Simulation finished. Starting animation...")
    
    animate_simulation(states, potential, DT, hamiltonian, initial_vector)
    print("Animation window closed.")