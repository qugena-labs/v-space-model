
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT, Diagonal

# --- Parameters ---
n_qubits = 10
grid_size = 2**n_qubits
L = 20.0  # System size from -L to L
dx = 2 * L / grid_size
x_values = np.linspace(-L, L, grid_size)

# Time evolution parameters
t_final = 10
n_steps = 50
dt = t_final / n_steps

# --- Potential and Coupling ---
V0 = 5.0
well_width = L / 2
potential = np.zeros(grid_size)
potential[np.abs(x_values) < well_width / 2] = -V0

# Continuity coupling parameter (g > 0 for attractive)
g = 0.0 
continuity_potential = np.zeros(grid_size)
mid_point_idx = grid_size // 2
continuity_potential[mid_point_idx -1] = -g
continuity_potential[mid_point_idx] = -g

total_potential = potential + continuity_potential

# --- Initial State ---
def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Wave packet in the right well
psi1 = gaussian(x_values, well_width / 4, L / 20)
# Wave packet in the left free space
psi2 = gaussian(x_values, -3 * L / 4, L / 20)

initial_state = psi1 + psi2
initial_state /= np.linalg.norm(initial_state)

# --- Log File ---
log_file = open("boundary_log.txt", "w")
log_file.write("Time,Boundary_Left_Prob,Boundary_Right_Prob\n")

# --- Quantum Circuit for Time Evolution ---
def time_evolution_circuit(state, dt):
    qc = QuantumCircuit(n_qubits)
    qc.initialize(state, qc.qubits)

    # Potential term
    potential_diagonals = np.exp(-1j * total_potential * dt)
    qc.append(Diagonal(potential_diagonals), range(n_qubits))

    # Kinetic term
    qc.append(QFT(n_qubits, do_swaps=True), range(n_qubits))
    momenta = np.fft.fftshift(np.fft.fftfreq(grid_size, d=dx)) * 2 * np.pi
    kinetic_diagonals = np.exp(-1j * 0.5 * momenta**2 * dt)
    qc.append(Diagonal(np.fft.ifftshift(kinetic_diagonals)), range(n_qubits))
    qc.append(QFT(n_qubits, do_swaps=True, inverse=True), range(n_qubits))

    return qc

# --- Simulation and Visualization ---
backend = Aer.get_backend('statevector_simulator')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

def update(frame):
    global initial_state
    ax1.clear()
    ax2.clear()

    probs = np.abs(initial_state)**2

    # --- Left Region Plot ---
    mid_point = grid_size // 2
    ax1.plot(x_values[:mid_point], probs[:mid_point])
    ax1.set_title(f'Left Region (-2x to 0) (t={frame*dt:.2f})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability')

    # --- Right Region Plot ---
    ax2.plot(x_values[mid_point:], probs[mid_point:])
    ax2.set_title(f'Right Region (0 to 2x) (t={frame*dt:.2f})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability')

    # --- Logging ---
    boundary_left_idx = mid_point - 1
    boundary_right_idx = mid_point
    log_file.write(f"{frame*dt:.2f},{probs[boundary_left_idx]:.6f},{probs[boundary_right_idx]:.6f}\n")

    # --- Evolve State ---
    if frame < n_steps - 1:
        qc_evol = time_evolution_circuit(initial_state, dt)
        transpiled_qc_evol = transpile(qc_evol, backend)
        result_evol = backend.run(transpiled_qc_evol).result()
        initial_state = result_evol.get_statevector()

ani = FuncAnimation(fig, update, frames=n_steps, blit=False)
plt.show()
log_file.close()
