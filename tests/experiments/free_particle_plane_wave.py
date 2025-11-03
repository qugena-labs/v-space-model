import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT, Diagonal

# --- Parameters ---
n_qubits_per_dim = 4
n_qubits = 2 * n_qubits_per_dim
grid_size = 2**n_qubits_per_dim

# Time evolution parameters
t_final = 5
n_steps = 20
dt = t_final / n_steps

# --- Initial State (Plane Wave) ---
# A plane wave is a state with a definite momentum.
# We create it by preparing a state with a single spike in the momentum basis,
# and then applying an inverse QFT.

momentum_x = 5
momentum_y = 2

qc_initial = QuantumCircuit(n_qubits)

# Prepare the momentum state |p_x>|p_y>
px_binary = format(momentum_x, f'0{n_qubits_per_dim}b')
py_binary = format(momentum_y, f'0{n_qubits_per_dim}b')

for i, bit in enumerate(reversed(px_binary)):
    if bit == '1':
        qc_initial.x(i)

for i, bit in enumerate(reversed(py_binary)):
    if bit == '1':
        qc_initial.x(i + n_qubits_per_dim)

# Inverse QFT to get the plane wave in position basis
qc_initial.append(QFT(n_qubits, do_swaps=True, inverse=True), range(n_qubits))

backend = Aer.get_backend('statevector_simulator')
transpiled_qc_initial = transpile(qc_initial, backend)
result_initial = backend.run(transpiled_qc_initial).result()
initial_state = result_initial.get_statevector()


# --- Hamiltonian (Free Particle) ---
# For a free particle, the potential is zero.
def time_evolution_circuit(state, dt):
    qc = QuantumCircuit(n_qubits)
    qc.initialize(state, qc.qubits)

    # Kinetic term (in momentum basis)
    qc.append(QFT(n_qubits, do_swaps=True), range(n_qubits))

    kinetic_diagonals = np.zeros(2**n_qubits, dtype=complex)
    momenta = np.fft.fftshift(np.arange(grid_size))
    for p_x_idx, p_x in enumerate(momenta):
        for p_y_idx, p_y in enumerate(momenta):
            kinetic_energy = 0.5 * (p_x**2 + p_y**2)
            phase = -kinetic_energy * dt
            index = p_x_idx * grid_size + p_y_idx
            kinetic_diagonals[index] = np.exp(1j * phase)
    qc.append(Diagonal(kinetic_diagonals), range(n_qubits))

    qc.append(QFT(n_qubits, do_swaps=True, inverse=True), range(n_qubits))

    return qc

# --- Simulation and Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

def update(frame):
    global initial_state
    ax1.clear()
    ax2.clear()

    # --- Position Space ---
    position_probs = np.abs(initial_state) ** 2
    position_grid = position_probs.reshape((grid_size, grid_size))
    exp_x = np.sum(position_grid * np.arange(grid_size)[:, np.newaxis])
    exp_y = np.sum(position_grid * np.arange(grid_size)[np.newaxis, :])

    ax1.imshow(position_grid, cmap='viridis', origin='lower')
    ax1.plot(exp_y, exp_x, 'ro')
    ax1.set_title(f'Position Space (t={frame*dt:.2f})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # --- Momentum Space ---
    qc_mom = QuantumCircuit(n_qubits)
    qc_mom.initialize(initial_state, qc_mom.qubits)
    qc_mom.append(QFT(n_qubits, do_swaps=True), range(n_qubits))
    transpiled_qc_mom = transpile(qc_mom, backend)
    result_mom = backend.run(transpiled_qc_mom).result()
    momentum_statevector = result_mom.get_statevector()
    momentum_probs = np.abs(momentum_statevector) ** 2
    momentum_grid = np.fft.fftshift(momentum_probs.reshape((grid_size, grid_size)))
    
    exp_px = np.sum(momentum_grid * np.arange(grid_size)[:, np.newaxis])
    exp_py = np.sum(momentum_grid * np.arange(grid_size)[np.newaxis, :])

    ax2.imshow(momentum_grid, cmap='viridis', origin='lower')
    ax2.plot(exp_py, exp_px, 'ro')
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