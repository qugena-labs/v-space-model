"""
Two-Layer Drainage Potential System
====================================

Quantum wave packet simulation with two potential layers:
- Top layer (surface): straight line with a hole
- Bottom layer (drainage): straight line below
- Wave transfers through the hole with momentum conservation

Based on test_potential_sys.py quantum mechanics approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.compiler import transpile
from qiskit_aer import Aer
from scipy.linalg import expm


def create_two_layer_potential(num_sites, hole_start, hole_end,
                                surface_height=10.0, drainage_height=0.0):
    """
    Creates a two-layer potential system.

    Surface layer: high potential with a hole
    Drainage layer: low potential below

    The hole allows coupling between layers.
    """
    # Total system: 2 * num_sites (top layer + bottom layer)
    total_sites = 2 * num_sites
    potential = np.zeros(total_sites)

    # Top layer (surface) - indices 0 to num_sites-1
    potential[:num_sites] = surface_height
    # Create hole in surface (no barrier)
    potential[hole_start:hole_end] = drainage_height

    # Bottom layer (drainage) - indices num_sites to 2*num_sites-1
    potential[num_sites:] = drainage_height

    return potential


def create_two_layer_hamiltonian(num_sites, potential, hole_start, hole_end,
                                  hopping_strength=1.0, vertical_coupling=0.3):
    """
    Creates Hamiltonian for two-layer system with coupling through hole.

    - Horizontal hopping within each layer
    - Vertical coupling at the hole location
    """
    total_sites = 2 * num_sites
    hamiltonian = np.zeros((total_sites, total_sites))

    # Fill diagonal with potential energies
    for i in range(total_sites):
        hamiltonian[i, i] = potential[i] + 2 * hopping_strength

    # Horizontal hopping in TOP layer (0 to num_sites-1)
    for i in range(num_sites - 1):
        hamiltonian[i, i + 1] = -hopping_strength
        hamiltonian[i + 1, i] = -hopping_strength

    # Horizontal hopping in BOTTOM layer (num_sites to 2*num_sites-1)
    for i in range(num_sites, total_sites - 1):
        hamiltonian[i, i + 1] = -hopping_strength
        hamiltonian[i + 1, i] = -hopping_strength

    # VERTICAL COUPLING through hole (top ↔ bottom)
    for i in range(hole_start, hole_end):
        top_idx = i
        bottom_idx = i + num_sites
        hamiltonian[top_idx, bottom_idx] = -vertical_coupling
        hamiltonian[bottom_idx, top_idx] = -vertical_coupling

    return hamiltonian


def create_initial_wave_packet(num_sites, position, width, momentum=0.0, layer='top'):
    """Creates Gaussian wave packet in specified layer."""
    total_sites = 2 * num_sites

    # Create wave packet
    if layer == 'top':
        # Top layer: indices 0 to num_sites-1
        x = np.arange(num_sites)
        gaussian = np.exp(-(x - position)**2 / (2 * width**2))
        momentum_phase = np.exp(1j * momentum * x)
        wave_packet_layer = gaussian * momentum_phase

        # Full state vector (top + bottom)
        wave_packet = np.zeros(total_sites, dtype=complex)
        wave_packet[:num_sites] = wave_packet_layer

    else:  # layer == 'bottom'
        # Bottom layer: indices num_sites to 2*num_sites-1
        x = np.arange(num_sites)
        gaussian = np.exp(-(x - position)**2 / (2 * width**2))
        momentum_phase = np.exp(1j * momentum * x)
        wave_packet_layer = gaussian * momentum_phase

        # Full state vector
        wave_packet = np.zeros(total_sites, dtype=complex)
        wave_packet[num_sites:] = wave_packet_layer

    wave_packet /= np.linalg.norm(wave_packet)  # Normalize

    # Create quantum circuit
    num_qubits = int(np.log2(total_sites))
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)
    qc.initialize(wave_packet, qr)

    return qc


def run_two_layer_simulation(hamiltonian, initial_state, time_steps, dt):
    """Simulates quantum time evolution in two-layer system."""
    backend = Aer.get_backend('statevector_simulator')
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


def animate_two_layer_simulation(states_over_time, num_sites, hole_start, hole_end,
                                  surface_height, drainage_height, dt):
    """Animates wave packet evolution in two-layer system."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(num_sites)

    # Visualization parameters
    max_prob = 0
    for state in states_over_time:
        prob_top = np.abs(state[:num_sites])**2
        prob_bottom = np.abs(state[num_sites:])**2
        max_prob = max(max_prob, np.max(prob_top), np.max(prob_bottom))

    prob_scale = 3.0 / max_prob if max_prob > 0 else 1.0

    def init():
        return []

    def update(frame):
        ax1.clear()
        ax2.clear()

        state = states_over_time[frame]

        # Top layer probability
        prob_top = np.abs(state[:num_sites])**2
        # Bottom layer probability
        prob_bottom = np.abs(state[num_sites:])**2

        # ===== PLOT 1: TOP LAYER (SURFACE) =====
        # Draw surface potential
        surface_potential = np.ones(num_sites) * surface_height
        surface_potential[hole_start:hole_end] = drainage_height

        ax1.fill_between(x, 0, surface_potential, alpha=0.3, color='tan', label='Surface Layer')
        ax1.plot(x, surface_potential, 'k-', linewidth=2, label='V(x) Surface')

        # Mark hole
        ax1.axvspan(hole_start, hole_end, alpha=0.3, color='red', label='Drainage Hole')

        # Plot wave packet probability
        wave_height = surface_potential + prob_scale * prob_top
        ax1.fill_between(x, surface_potential, wave_height, alpha=0.7, color='blue', label='Wave Packet')
        ax1.plot(x, wave_height, 'b-', linewidth=1.5)

        ax1.set_xlabel('Position x', fontsize=12)
        ax1.set_ylabel('V(x) - Potential', fontsize=12)
        ax1.set_title(f'TOP LAYER (SURFACE) | t={frame * dt:.2f}', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, num_sites)
        ax1.set_ylim(-1, surface_height + 5)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # ===== PLOT 2: BOTTOM LAYER (DRAINAGE) =====
        # Draw drainage potential
        drainage_potential = np.ones(num_sites) * drainage_height

        ax2.fill_between(x, -3, drainage_potential, alpha=0.3, color='lightgray', label='Hollow Space')
        ax2.plot(x, drainage_potential, 'k-', linewidth=2, label='V(x) Drainage')

        # Mark hole region
        ax2.axvspan(hole_start, hole_end, alpha=0.3, color='red', label='Coupling Region')

        # Plot wave packet probability
        wave_height_drain = drainage_potential + prob_scale * prob_bottom
        ax2.fill_between(x, drainage_potential, wave_height_drain, alpha=0.7, color='cyan', label='Wave Packet')
        ax2.plot(x, wave_height_drain, 'c-', linewidth=1.5)

        ax2.set_xlabel('Position x', fontsize=12)
        ax2.set_ylabel('V(x) - Potential', fontsize=12)
        ax2.set_title(f'BOTTOM LAYER (DRAINAGE) | t={frame * dt:.2f}', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, num_sites)
        ax2.set_ylim(-3, drainage_height + 5)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Statistics
        total_top = np.sum(prob_top)
        total_bottom = np.sum(prob_bottom)
        stats_text = f'Surface: {total_top:.3f}\nDrainage: {total_bottom:.3f}\nTotal: {total_top + total_bottom:.3f}'
        ax2.text(0.02, 0.98, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        return []

    ani = animation.FuncAnimation(fig, update, frames=len(states_over_time),
                                 init_func=init, blit=False, interval=50)
    plt.tight_layout()
    plt.show()


def main():
    print("\n" + "="*70)
    print("TWO-LAYER DRAINAGE QUANTUM WAVE SIMULATION")
    print("="*70)

    # Simulation parameters
    NUM_QUBITS = 7
    NUM_SITES = 2**NUM_QUBITS  # 128 sites per layer
    TOTAL_SITES = 2 * NUM_SITES  # 256 total (top + bottom)

    print(f"  Sites per layer: {NUM_SITES}")
    print(f"  Total sites: {TOTAL_SITES}")
    print(f"  Qubits: {NUM_QUBITS + 1}")

    # Potential parameters
    SURFACE_HEIGHT = 10.0
    DRAINAGE_HEIGHT = 0.0
    HOLE_START = int(0.35 * NUM_SITES)  # Hole at 35-45%
    HOLE_END = int(0.45 * NUM_SITES)

    print(f"  Surface potential: {SURFACE_HEIGHT}")
    print(f"  Drainage potential: {DRAINAGE_HEIGHT}")
    print(f"  Hole location: indices {HOLE_START} to {HOLE_END}")

    # Initial wave packet (on surface, moving right)
    INITIAL_POSITION = int(0.15 * NUM_SITES)  # Start at 15%
    INITIAL_WIDTH = 8
    INITIAL_MOMENTUM = 0.4  # Rightward momentum

    print(f"  Initial wave position: {INITIAL_POSITION}")
    print(f"  Initial momentum: {INITIAL_MOMENTUM}")

    # Time evolution
    TIME_STEPS = 500
    DT = 0.6

    print(f"  Time steps: {TIME_STEPS}")
    print(f"  dt: {DT}")
    print("="*70 + "\n")

    # Create system
    print("Creating two-layer potential system...")
    potential = create_two_layer_potential(NUM_SITES, HOLE_START, HOLE_END,
                                           SURFACE_HEIGHT, DRAINAGE_HEIGHT)

    print("Creating Hamiltonian with vertical coupling...")
    hamiltonian = create_two_layer_hamiltonian(NUM_SITES, potential,
                                                HOLE_START, HOLE_END,
                                                hopping_strength=1.0,
                                                vertical_coupling=0.5)

    print("Initializing wave packet on surface...")
    initial_state = create_initial_wave_packet(NUM_SITES, INITIAL_POSITION,
                                                INITIAL_WIDTH, INITIAL_MOMENTUM,
                                                layer='top')

    print("Running quantum simulation...")
    states, initial_vector = run_two_layer_simulation(hamiltonian, initial_state,
                                                       TIME_STEPS, DT)

    print("Starting animation...")
    animate_two_layer_simulation(states, NUM_SITES, HOLE_START, HOLE_END,
                                  SURFACE_HEIGHT, DRAINAGE_HEIGHT, DT)

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
