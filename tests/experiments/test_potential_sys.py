
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.compiler import transpile
from qiskit_aer import Aer

def create_potential_landscape(num_sites, steps):
    """Creates a two-layer potential energy landscape with drainage hole.

    Top layer: V(x) = 0 (surface/ground level) with a hole
    Bottom layer: V(x) = -V (negative, lower energy - drainage)
    """
    potential = np.zeros(num_sites)

    # SURFACE LAYER: V(x) = 0 (this is the ground, where wave sits on top)
    # potential is already zeros, so surface is at V = 0

    # DRAINAGE WELL: Deep potential well to trap the wave
    hole_start = int(0.35 * num_sites)  # Hole at 35-50%
    hole_end = int(0.50 * num_sites)

    # At the hole, drop down to drainage potential - MUCH DEEPER to trap wave
    drainage_depth = -5.0  # Deep negative potential well (strong attraction)
    potential[hole_start:hole_end] = drainage_depth

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
    """Creates a Gaussian wave packet as the initial state."""
    x = np.arange(num_sites)
    gaussian = np.exp(-(x - position)**2 / (2 * width**2))
    momentum_phase = np.exp(1j * momentum * x)
    wave_packet = gaussian * momentum_phase
    wave_packet /= np.linalg.norm(wave_packet)  # Normalize
    
    qr = QuantumRegister(int(np.log2(num_sites)))
    qc = QuantumCircuit(qr)
    qc.initialize(wave_packet, qr)
    return qc

def run_simulation(hamiltonian, initial_state, time_steps, dt):
    """Simulates the time evolution of the system."""
    from scipy.linalg import expm
    num_sites = hamiltonian.shape[0]
    num_qubits = int(np.log2(num_sites))
    
    # Get the initial state vector
    backend = Aer.get_backend('statevector_simulator')
    
    # Ensure the initial_state circuit is transpiled for the simulator
    # This is good practice although for statevector_simulator it might not be strictly necessary
    t_initial_state = transpile(initial_state, backend)
    
    initial_vector = Statevector(t_initial_state).data

    states_over_time = []
    current_state = initial_vector
    
    # Time evolution operator
    U = expm(-1j * hamiltonian * dt)
    
    for _ in range(time_steps):
        current_state = U @ current_state
        states_over_time.append(current_state)
        
    return states_over_time, initial_vector

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
    
    # Scale the probability density so its peak is at a specific energy level for visualization
    max_initial_prob = np.max(np.abs(states_over_time[0])**2)
    # Set the visual peak height to be just above the first potential step
    visual_peak_height = 0.2
    prob_scaling_factor = visual_peak_height / max_initial_prob if max_initial_prob > 0 else 1

    line, = ax.plot(x, prob_scaling_factor * np.abs(states_over_time[0])**2, label="Wave Packet Probability")
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.set_xlabel("Position x")
    ax.set_ylabel("V(x) - Potential Energy")
    ax.set_title("Wave ON SURFACE (V=0) FALLS INTO DEEP DRAINAGE WELL (V=-5.0)")
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, label='Ground Level (V=0)')

    def update(frame):
        line.set_ydata(prob_scaling_factor * np.abs(states_over_time[frame])**2)
        time_text.set_text(f"Time = {frame * dt:.2f}")
        return line, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(states_over_time), blit=True, interval=50)
    plt.show()

if __name__ == '__main__':
    # --- Simulation Parameters ---
    NUM_QUBITS = 8
    NUM_SITES = 2**NUM_QUBITS

    # Two-layer drainage system (no steps parameter needed anymore)
    # Hole at 35-45% of domain

    # Initial wave packet parameters (starts on the left)
    INITIAL_POSITION = int(0.15 * NUM_SITES)  # Start at 15%
    INITIAL_WIDTH = 10
    INITIAL_MOMENTUM = 0.45  # This gives the wave packet a rightward "kick"

    # Time evolution parameters
    TIME_STEPS = 600
    DT = 0.5  # Time step size

    # --- Setup and Run ---
    print("\n" + "="*70)
    print("DRAINAGE QUANTUM WAVE SIMULATION")
    print("="*70)
    print(f"  Sites: {NUM_SITES}")
    print(f"  Surface level: V = 0 (ground)")
    print(f"  Drainage well depth: V = -5.0 (DEEP POTENTIAL WELL)")
    print(f"  Well location: 35-50% (opening to drainage)")
    print(f"  Initial position: {INITIAL_POSITION}")
    print(f"  Initial momentum: {INITIAL_MOMENTUM}")
    print("  Wave sits ON TOP of surface, FALLS and GETS TRAPPED in drainage well")
    print("="*70 + "\n")

    potential = create_potential_landscape(NUM_SITES, None)  # steps parameter unused now
    hamiltonian = create_hamiltonian(NUM_SITES, potential)
    initial_state_circuit = create_initial_state(
        NUM_SITES, INITIAL_POSITION, INITIAL_WIDTH, INITIAL_MOMENTUM
    )

    print("Running simulation...")
    states, initial_vector = run_simulation(hamiltonian, initial_state_circuit, TIME_STEPS, DT)
    print("Simulation finished. Starting animation...")

    animate_simulation(states, potential, DT, hamiltonian, initial_vector)
    print("\nAnimation window closed.")
