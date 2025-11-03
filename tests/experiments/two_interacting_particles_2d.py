import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT, Diagonal

# --- Parameters ---
n_qubits_per_dim = 3  # 3 qubits for each dimension -> 8x8 grid
n_particles = 2
n_qubits = n_qubits_per_dim * 2 * n_particles
grid_size = 2**n_qubits_per_dim

# Time evolution parameters
t_final = 5
n_steps = 20
dt = t_final / n_steps

# --- Hamiltonian ---
# Potential (Harmonic Oscillator for each particle)
def potential(x1, y1, x2, y2, k=0.5):
    center = grid_size / 2
    v1 = 0.5 * k * ((x1 - center)**2 + (y1 - center)**2)
    v2 = 0.5 * k * ((x2 - center)**2 + (y2 - center)**2)
    return v1 + v2

# Interaction Potential
def interaction_potential(x1, y1, x2, y2, g=0.1):
    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
    if dist_sq < 1e-6:
        return g * 100 # Avoid singularity
    return g / np.sqrt(dist_sq)

# --- Initial State ---
def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

initial_state = np.zeros(2**n_qubits, dtype=complex)
for i1 in range(grid_size):
    for j1 in range(grid_size):
        for i2 in range(grid_size):
            for j2 in range(grid_size):
                amp1 = gaussian(i1, grid_size / 4, grid_size / 16) * gaussian(j1, grid_size / 4, grid_size / 16)
                amp2 = gaussian(i2, 3 * grid_size / 4, grid_size / 16) * gaussian(j2, 3 * grid_size / 4, grid_size / 16)
                index = i1 * (grid_size**3) + j1 * (grid_size**2) + i2 * grid_size + j2
                initial_state[index] = amp1 * amp2
initial_state /= np.linalg.norm(initial_state)

# --- Quantum Circuit for Time Evolution ---
def time_evolution_circuit(state, dt):
    qc = QuantumCircuit(n_qubits)
    qc.initialize(state, qc.qubits)

    # Potential terms
    potential_diagonals = np.zeros(2**n_qubits, dtype=complex)
    for i1 in range(grid_size):
        for j1 in range(grid_size):
            for i2 in range(grid_size):
                for j2 in range(grid_size):
                    v = potential(i1, j1, i2, j2) + interaction_potential(i1, j1, i2, j2)
                    phase = -v * dt
                    index = i1 * (grid_size**3) + j1 * (grid_size**2) + i2 * grid_size + j2
                    potential_diagonals[index] = np.exp(1j * phase)
    qc.append(Diagonal(potential_diagonals), range(n_qubits))

    # Kinetic term (via QFT)
    for particle in range(n_particles):
        q_indices = range(particle * n_qubits_per_dim * 2, (particle + 1) * n_qubits_per_dim * 2)
        qc.append(QFT(len(q_indices), do_swaps=True), q_indices)

    kinetic_diagonals = np.zeros(2**n_qubits, dtype=complex)
    for p1x in range(grid_size):
        for p1y in range(grid_size):
            for p2x in range(grid_size):
                for p2y in range(grid_size):
                    ke = 0.5 * (p1x**2 + p1y**2) + 0.5 * (p2x**2 + p2y**2)
                    phase = -ke * dt
                    index = p1x * (grid_size**3) + p1y * (grid_size**2) + p2x * grid_size + p2y
                    kinetic_diagonals[index] = np.exp(1j * phase)
    qc.append(Diagonal(kinetic_diagonals), range(n_qubits))

    for particle in range(n_particles):
        q_indices = range(particle * n_qubits_per_dim * 2, (particle + 1) * n_qubits_per_dim * 2)
        qc.append(QFT(len(q_indices), do_swaps=True, inverse=True), q_indices)

    return qc

# --- Simulation and Visualization ---
backend = Aer.get_backend('statevector_simulator')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

def update(frame):
    global initial_state
    ax1.clear()
    ax2.clear()

    # --- Position Space ---
    probs_4d = np.abs(initial_state)**2
    probs_p1 = probs_4d.reshape(grid_size, grid_size, grid_size, grid_size).sum(axis=(2, 3))
    probs_p2 = probs_4d.reshape(grid_size, grid_size, grid_size, grid_size).sum(axis=(0, 1))

    exp_x1 = np.sum(probs_p1 * np.arange(grid_size)[:, np.newaxis])
    exp_y1 = np.sum(probs_p1 * np.arange(grid_size)[np.newaxis, :])
    exp_x2 = np.sum(probs_p2 * np.arange(grid_size)[:, np.newaxis])
    exp_y2 = np.sum(probs_p2 * np.arange(grid_size)[np.newaxis, :])

    ax1.imshow(probs_p1 + probs_p2, cmap='viridis', origin='lower')
    ax1.plot(exp_y1, exp_x1, 'ro')
    ax1.plot(exp_y2, exp_x2, 'bo')
    ax1.set_title(f'Position Space (t={frame*dt:.2f})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # --- Momentum Space ---
    qc_mom = QuantumCircuit(n_qubits)
    qc_mom.initialize(initial_state, qc_mom.qubits)
    for p in range(n_particles):
        q_indices = range(p * n_qubits_per_dim * 2, (p + 1) * n_qubits_per_dim * 2)
        qc_mom.append(QFT(len(q_indices), do_swaps=True), q_indices)

    transpiled_qc_mom = transpile(qc_mom, backend)
    result_mom = backend.run(transpiled_qc_mom).result()
    momentum_statevector = result_mom.get_statevector()
    mom_probs_4d = np.abs(momentum_statevector)**2
    mom_probs_p1 = mom_probs_4d.reshape(grid_size, grid_size, grid_size, grid_size).sum(axis=(2, 3))
    mom_probs_p2 = mom_probs_4d.reshape(grid_size, grid_size, grid_size, grid_size).sum(axis=(0, 1))

    exp_px1 = np.sum(mom_probs_p1 * np.arange(grid_size)[:, np.newaxis])
    exp_py1 = np.sum(mom_probs_p1 * np.arange(grid_size)[np.newaxis, :])
    exp_px2 = np.sum(mom_probs_p2 * np.arange(grid_size)[:, np.newaxis])
    exp_py2 = np.sum(mom_probs_p2 * np.arange(grid_size)[np.newaxis, :])

    ax2.imshow(mom_probs_p1 + mom_probs_p2, cmap='viridis', origin='lower')
    ax2.plot(exp_py1, exp_px1, 'ro')
    ax2.plot(exp_py2, exp_px2, 'bo')
    ax2.set_title(f'Momentum Space (t={frame*dt:.2f})')
    ax2.set_xlabel('p_x')
    ax2.set_ylabel('p_y')

    # --- Evolve State ---
    if frame < n_steps - 1:
        qc_evol = time_evolution_circuit(initial_state, dt)
        transpiled_qc_evol = transpile(qc_evol, backend)
        result_evol = backend.run(transpiled_qc_evol).result()
        initial_state = result_evol.get_statevector()

ani = FuncAnimation(fig, update, frames=n_steps, blit=False)
plt.show()