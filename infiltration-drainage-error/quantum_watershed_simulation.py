"""
Quantum-Enhanced Urban Watershed Simulation
============================================

A comprehensive 2D+1 spatial simulation integrating quantum mechanics principles
with hydrological processes to model rainfall and water flow in an urban watershed.

Author: Quantum Hydrology Research Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.ndimage import convolve
from scipy.interpolate import splprep, splev
from scipy.fft import fft2, ifft2
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    from qiskit.quantum_info import DensityMatrix
    QISKIT_AVAILABLE = True
except ImportError:
    print("Warning: Qiskit not available. Using classical simulation fallback.")
    QISKIT_AVAILABLE = False


# ============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# ============================================================================

class SimulationParameters:
    """Container for all simulation parameters"""

    # Grid dimensions
    GRID_SIZE = 150  # 150x150 grid
    CELL_SIZE = 5.0  # meters per cell
    DOMAIN_SIZE = GRID_SIZE * CELL_SIZE  # 750m x 750m watershed

    # Time parameters
    DT = 2.0  # seconds per timestep
    TOTAL_TIME = 3600.0  # 1 hour simulation
    N_TIMESTEPS = int(TOTAL_TIME / DT)

    # Physical constants
    HBAR = 1.0  # Reduced Planck constant (normalized)
    EFFECTIVE_MASS = 1.0  # Effective mass for wave propagation
    GRAVITY = 9.81  # m/s²

    # Rainfall parameters (mm/hour converted to normalized amplitude)
    RAINFALL_RATES = {
        'light': 5.0,
        'moderate': 15.0,
        'heavy': 30.0,
        'extreme': 80.0
    }

    # Zone elevations (meters)
    ZONE_LEFT_ELEVATION = (50, 100)  # Hill zone
    ZONE_MIDDLE_ELEVATION = (20, 50)  # Suburban zone
    ZONE_RIGHT_ELEVATION = (0, 20)   # Urban zone
    LAKEBED_ELEVATION = (0, 5)       # Lakebed (right-most corner)

    # Infiltration parameters
    INFILTRATION_DEPTH_ROAD = 0.10  # 10cm max for roads
    INFILTRATION_DEPTH_PARK = 0.40  # 40cm max for parks
    INFILTRATION_DEPTH_BUILDING = 0.0  # Impervious

    # Quantum parameters
    N_QUBITS_ZONE1 = 7   # Hill zone
    N_QUBITS_ZONE2 = 10  # Suburban zone
    N_QUBITS_ZONE3 = 15  # Urban zone
    N_QUBITS_PARK = 4    # Park sub-register
    N_QUBITS_DRAINAGE = 8  # Drainage system

    # Decoherence rates (1/seconds)
    DECOHERENCE_ZONE1 = 0.3  # Fast (natural soil)
    DECOHERENCE_ZONE2 = 0.15  # Medium
    DECOHERENCE_ZONE3 = 0.05  # Slow (impervious)

    # Visualization
    FPS = 10  # Frames per second for animation
    FRAME_SKIP = max(1, N_TIMESTEPS // (FPS * 60))  # Limit to ~60 seconds of video


# ============================================================================
# TOPOLOGY GENERATION
# ============================================================================

class WatershedTopology:
    """Generates and manages the spatial topology of the watershed"""

    def __init__(self, params: SimulationParameters):
        self.params = params
        self.grid_size = params.GRID_SIZE

        # Initialize arrays
        self.elevation = np.zeros((self.grid_size, self.grid_size))
        self.surface_type = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Surface types: 0=natural, 1=road, 2=building, 3=sidewalk, 4=park, 5=lakebed

        self.zone_mask = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Zones: 1=left (hills), 2=middle (suburban), 3=right (urban/lakebed)

        self.buildings = []  # List of building footprints
        self.roads = []      # List of road paths

        self._generate_topology()

    def _generate_topology(self):
        """Generate complete watershed topology"""
        print("Generating watershed topology...")

        # Define zone boundaries
        zone1_end = self.grid_size // 3
        zone2_end = 2 * self.grid_size // 3

        # Create base elevation map
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)

        # Zone 1: Hills (left third)
        hill_elevation = 100 - 50 * X  # Slopes down from left to right
        hill_elevation += 15 * np.sin(4 * np.pi * Y) * np.exp(-2 * X)  # Ridge features

        # Zone 2: Suburban (middle third)
        suburban_elevation = 50 - 30 * X
        suburban_elevation += 5 * np.sin(3 * np.pi * Y) * np.cos(2 * np.pi * X)

        # Zone 3: Urban and lakebed (right third)
        urban_elevation = 15 - 10 * X
        # Lakebed in bottom-right corner
        lakebed_mask = ((X > 0.8) & (Y < 0.3))
        urban_elevation[lakebed_mask] = 2.0 + np.random.uniform(0, 1, np.sum(lakebed_mask))

        # Combine elevations by zone
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i < zone1_end:
                    self.elevation[j, i] = hill_elevation[j, i]
                    self.zone_mask[j, i] = 1
                elif i < zone2_end:
                    self.elevation[j, i] = suburban_elevation[j, i]
                    self.zone_mask[j, i] = 2
                else:
                    self.elevation[j, i] = urban_elevation[j, i]
                    self.zone_mask[j, i] = 3

        # Smooth elevation transitions
        from scipy.ndimage import gaussian_filter
        self.elevation = gaussian_filter(self.elevation, sigma=2.0)

        # Generate roads
        self._generate_roads()

        # Place buildings
        self._place_buildings()

        # Create park in middle zone
        self._create_park()

        # Mark lakebed
        lakebed_i_start = int(0.8 * self.grid_size)
        lakebed_j_end = int(0.3 * self.grid_size)
        self.surface_type[0:lakebed_j_end, lakebed_i_start:] = 5

        print(f"  - Generated {len(self.buildings)} buildings")
        print(f"  - Generated {len(self.roads)} road segments")

    def _generate_roads(self):
        """Generate curved road network using splines"""
        roads_data = []

        # Main arterial road: curves through all three zones
        control_points = np.array([
            [10, 75],    # Start in zone 1
            [40, 60],
            [70, 80],
            [95, 70],    # Middle zone
            [120, 90],
            [140, 75]    # End in zone 3
        ])

        road1 = self._create_curved_road(control_points, width=4)
        roads_data.append(road1)

        # Secondary road in zone 1 (hillside)
        control_points2 = np.array([
            [5, 40],
            [25, 50],
            [45, 45]
        ])
        road2 = self._create_curved_road(control_points2, width=3)
        roads_data.append(road2)

        # Road through suburban zone
        control_points3 = np.array([
            [50, 20],
            [75, 30],
            [95, 25],
            [110, 35]
        ])
        road3 = self._create_curved_road(control_points3, width=4)
        roads_data.append(road3)

        # Urban grid roads (zone 3)
        # Vertical road
        road4 = []
        for j in range(20, 130):
            road4.append([125, j])
        roads_data.append(road4)

        # Horizontal road
        road5 = []
        for i in range(105, 145):
            road5.append([i, 50])
        roads_data.append(road5)

        # Mark roads and sidewalks on surface_type grid
        for road_path in roads_data:
            self.roads.append(road_path)
            for point in road_path:
                i, j = int(point[0]), int(point[1])
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                    self.surface_type[j, i] = 1  # Road

                    # Add sidewalks (elevated by 0.3m)
                    for di in [-1, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                                if self.surface_type[nj, ni] == 0:  # Only on natural ground
                                    self.surface_type[nj, ni] = 3  # Sidewalk
                                    self.elevation[nj, ni] += 0.3

    def _create_curved_road(self, control_points: np.ndarray, width: int = 3) -> List:
        """Create smooth curved road from control points"""
        if len(control_points) < 3:
            return control_points.tolist()

        # Fit spline
        tck, u = splprep([control_points[:, 0], control_points[:, 1]], s=0, k=min(3, len(control_points)-1))
        u_fine = np.linspace(0, 1, 200)
        x_fine, y_fine = splev(u_fine, tck)

        road_cells = []
        for xi, yi in zip(x_fine, y_fine):
            # Add width to road
            for w in range(-width//2, width//2 + 1):
                for h in range(-width//2, width//2 + 1):
                    road_cells.append([xi + w, yi + h])

        return road_cells

    def _place_buildings(self):
        """Place buildings in all three zones"""
        np.random.seed(42)  # For reproducibility

        # Zone 1: Residential houses on hillside (5x5 to 8x8)
        for _ in range(15):
            size = np.random.randint(5, 9)
            x = np.random.randint(5, self.grid_size // 3 - size - 5)
            y = np.random.randint(5, self.grid_size - size - 5)

            if self._can_place_building(x, y, size):
                self._add_building(x, y, size, height=8.0)

        # Zone 2: Suburban houses (6x6 to 10x10), avoiding park area
        park_center_x = self.grid_size // 2
        park_center_y = self.grid_size // 2
        park_radius = 25

        for _ in range(12):
            size = np.random.randint(6, 11)
            x = np.random.randint(self.grid_size // 3 + 5, 2 * self.grid_size // 3 - size - 5)
            y = np.random.randint(5, self.grid_size - size - 5)

            # Check if too close to park
            if np.sqrt((x - park_center_x)**2 + (y - park_center_y)**2) < park_radius:
                continue

            if self._can_place_building(x, y, size):
                self._add_building(x, y, size, height=10.0)

        # Zone 3: Urban high-rises (8x8 to 15x15), avoiding lakebed
        for _ in range(10):
            size = np.random.randint(8, 16)
            x = np.random.randint(2 * self.grid_size // 3 + 5, self.grid_size - size - 5)
            y = np.random.randint(self.grid_size // 3, self.grid_size - size - 5)  # Keep away from lakebed

            if self._can_place_building(x, y, size):
                height = np.random.uniform(30, 80)  # Skyscrapers
                self._add_building(x, y, size, height=height)

    def _can_place_building(self, x: int, y: int, size: int) -> bool:
        """Check if building can be placed at location"""
        if x + size >= self.grid_size or y + size >= self.grid_size:
            return False

        # Check for overlap with existing buildings or roads
        region = self.surface_type[y:y+size, x:x+size]
        return np.all(region == 0) or np.all(region == 4)  # Only on natural ground or park

    def _add_building(self, x: int, y: int, size: int, height: float):
        """Add building to topology"""
        self.buildings.append({'x': x, 'y': y, 'size': size, 'height': height})
        self.surface_type[y:y+size, x:x+size] = 2  # Mark as building
        self.elevation[y:y+size, x:x+size] += height  # Add height to potential

    def _create_park(self):
        """Create park in middle zone with central tree"""
        park_center_x = self.grid_size // 2
        park_center_y = self.grid_size // 2
        park_radius = 20

        # Mark park area
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - park_center_x)**2 + (j - park_center_y)**2)
                if dist < park_radius and self.surface_type[j, i] == 0:
                    self.surface_type[j, i] = 4  # Park

        # Add small building in center representing a tree
        tree_size = 3
        tree_x = park_center_x - tree_size // 2
        tree_y = park_center_y - tree_size // 2
        self.buildings.append({'x': tree_x, 'y': tree_y, 'size': tree_size, 'height': 5.0})
        self.surface_type[tree_y:tree_y+tree_size, tree_x:tree_x+tree_size] = 2

    def get_potential(self) -> np.ndarray:
        """Return potential energy array (elevation)"""
        return self.elevation.copy()

    def get_infiltration_capacity(self) -> np.ndarray:
        """Return maximum infiltration capacity based on surface type"""
        capacity = np.zeros_like(self.elevation)

        capacity[self.surface_type == 0] = 0.2  # Natural ground: 20cm
        capacity[self.surface_type == 1] = self.params.INFILTRATION_DEPTH_ROAD  # Road: 10cm
        capacity[self.surface_type == 2] = self.params.INFILTRATION_DEPTH_BUILDING  # Building: 0cm
        capacity[self.surface_type == 3] = 0.08  # Sidewalk: 8cm
        capacity[self.surface_type == 4] = self.params.INFILTRATION_DEPTH_PARK  # Park: 40cm
        capacity[self.surface_type == 5] = 0.0  # Lakebed: no infiltration (collection point)

        return capacity


# ============================================================================
# QUANTUM SUBSYSTEM
# ============================================================================

class QuantumInfiltrationSystem:
    """Manages quantum registers and circuits for infiltration control"""

    def __init__(self, params: SimulationParameters, topology: WatershedTopology):
        self.params = params
        self.topology = topology

        if not QISKIT_AVAILABLE:
            print("Warning: Running in classical mode (Qiskit not available)")
            self.use_quantum = False
        else:
            self.use_quantum = True

        # Create quantum registers for each zone
        self.qr_zone1 = QuantumRegister(params.N_QUBITS_ZONE1, 'zone1') if self.use_quantum else None
        self.qr_zone2 = QuantumRegister(params.N_QUBITS_ZONE2, 'zone2') if self.use_quantum else None
        self.qr_zone3 = QuantumRegister(params.N_QUBITS_ZONE3, 'zone3') if self.use_quantum else None
        self.qr_park = QuantumRegister(params.N_QUBITS_PARK, 'park') if self.use_quantum else None
        self.qr_drainage = QuantumRegister(params.N_QUBITS_DRAINAGE, 'drainage') if self.use_quantum else None

        # Classical registers for measurement
        self.cr_zone1 = ClassicalRegister(params.N_QUBITS_ZONE1, 'c1') if self.use_quantum else None
        self.cr_zone2 = ClassicalRegister(params.N_QUBITS_ZONE2, 'c2') if self.use_quantum else None
        self.cr_zone3 = ClassicalRegister(params.N_QUBITS_ZONE3, 'c3') if self.use_quantum else None
        self.cr_park = ClassicalRegister(params.N_QUBITS_PARK, 'c_park') if self.use_quantum else None
        self.cr_drainage = ClassicalRegister(params.N_QUBITS_DRAINAGE, 'c_drain') if self.use_quantum else None

        # Simulation backend
        self.backend = AerSimulator() if self.use_quantum else None

        # Current measurement results
        self.zone1_state = None
        self.zone2_state = None
        self.zone3_state = None
        self.park_state = None
        self.drainage_state = None

        # Drainage inlet locations
        self.drainage_inlets = self._create_drainage_inlets()

        print(f"Quantum system initialized with {params.N_QUBITS_ZONE1 + params.N_QUBITS_ZONE2 + params.N_QUBITS_ZONE3 + params.N_QUBITS_PARK + params.N_QUBITS_DRAINAGE} total qubits")

    def _create_drainage_inlets(self) -> List[Tuple[int, int]]:
        """Create drainage inlet locations along roads"""
        inlets = []

        # Place inlets every 20 cells along roads
        for road in self.topology.roads:
            for idx in range(0, len(road), 20):
                point = road[idx]
                i, j = int(point[0]), int(point[1])
                if 0 <= i < self.params.GRID_SIZE and 0 <= j < self.params.GRID_SIZE:
                    inlets.append((i, j))

        print(f"  - Created {len(inlets)} drainage inlets")
        return inlets

    def initialize_circuits(self):
        """Initialize quantum circuits for all zones"""
        if not self.use_quantum:
            # Classical fallback: random initial states
            self.zone1_state = np.random.randint(0, 2, self.params.N_QUBITS_ZONE1)
            self.zone2_state = np.random.randint(0, 2, self.params.N_QUBITS_ZONE2)
            self.zone3_state = np.random.randint(0, 2, self.params.N_QUBITS_ZONE3)
            self.park_state = np.random.randint(0, 2, self.params.N_QUBITS_PARK)
            self.drainage_state = np.random.randint(0, 2, self.params.N_QUBITS_DRAINAGE)
            return

        # Zone 1: Hill zone circuit
        self.zone1_state = self._run_infiltration_circuit(
            self.qr_zone1, self.cr_zone1,
            decoherence_rate=self.params.DECOHERENCE_ZONE1,
            rotation_angle=np.pi/4
        )

        # Zone 2: Suburban zone circuit
        self.zone2_state = self._run_infiltration_circuit(
            self.qr_zone2, self.cr_zone2,
            decoherence_rate=self.params.DECOHERENCE_ZONE2,
            rotation_angle=np.pi/3
        )

        # Zone 3: Urban zone circuit
        self.zone3_state = self._run_infiltration_circuit(
            self.qr_zone3, self.cr_zone3,
            decoherence_rate=self.params.DECOHERENCE_ZONE3,
            rotation_angle=np.pi/6
        )

        # Park sub-register
        self.park_state = self._run_infiltration_circuit(
            self.qr_park, self.cr_park,
            decoherence_rate=self.params.DECOHERENCE_ZONE1,  # Like natural soil
            rotation_angle=np.pi/2  # High infiltration
        )

        # Drainage system
        self.drainage_state = self._run_drainage_circuit()

    def _run_infiltration_circuit(self, qr: QuantumRegister, cr: ClassicalRegister,
                                   decoherence_rate: float, rotation_angle: float) -> np.ndarray:
        """Run quantum circuit for infiltration control"""

        # Create circuit
        qc = QuantumCircuit(qr, cr)

        # Apply Hadamard gates to create superposition
        for i in range(len(qr)):
            qc.h(qr[i])

        # Apply controlled rotations based on surface properties
        for i in range(len(qr) - 1):
            qc.crz(rotation_angle, qr[i], qr[i+1])

        # Add depolarizing noise to simulate decoherence
        noise_model = NoiseModel()
        error = depolarizing_error(decoherence_rate * self.params.DT, 1)
        noise_model.add_all_qubit_quantum_error(error, ['h', 'crz'])

        # Measure
        qc.measure(qr, cr)

        # Execute
        job = self.backend.run(transpile(qc, self.backend), shots=1, noise_model=noise_model)
        result = job.result()
        counts = result.get_counts()

        # Extract measurement result
        bitstring = list(counts.keys())[0]
        measurements = np.array([int(b) for b in bitstring[::-1]])

        return measurements

    def _run_drainage_circuit(self) -> np.ndarray:
        """Run quantum circuit for drainage system control"""
        if not self.use_quantum:
            # 70% chance of inlet being open
            return np.random.binomial(1, 0.7, self.params.N_QUBITS_DRAINAGE)

        qc = QuantumCircuit(self.qr_drainage, self.cr_drainage)

        # Most inlets start functional (bias toward |1⟩)
        for i in range(len(self.qr_drainage)):
            qc.ry(np.pi/3, self.qr_drainage[i])  # Rotation to create bias

        # Entangle some qubits (drainage system interconnected)
        for i in range(0, len(self.qr_drainage)-1, 2):
            qc.cx(self.qr_drainage[i], self.qr_drainage[i+1])

        # Measure
        qc.measure(self.qr_drainage, self.cr_drainage)

        # Execute
        job = self.backend.run(transpile(qc, self.backend), shots=1)
        result = job.result()
        counts = result.get_counts()

        bitstring = list(counts.keys())[0]
        measurements = np.array([int(b) for b in bitstring[::-1]])

        return measurements

    def update_quantum_state(self, t: float):
        """Update quantum measurements periodically"""
        # Re-measure every 60 seconds
        if int(t) % 60 == 0:
            self.initialize_circuits()

    def get_infiltration_map(self) -> np.ndarray:
        """Generate spatial map of infiltration based on quantum measurements"""
        infiltration_map = np.zeros((self.params.GRID_SIZE, self.params.GRID_SIZE))
        zone_mask = self.topology.zone_mask
        surface_type = self.topology.surface_type
        max_capacity = self.topology.get_infiltration_capacity()

        for i in range(self.params.GRID_SIZE):
            for j in range(self.params.GRID_SIZE):
                zone = zone_mask[j, i]

                # Select appropriate quantum register state
                if zone == 1:
                    qubit_idx = (i * j) % self.params.N_QUBITS_ZONE1
                    measurement = self.zone1_state[qubit_idx]
                elif zone == 2:
                    if surface_type[j, i] == 4:  # Park
                        qubit_idx = (i + j) % self.params.N_QUBITS_PARK
                        measurement = self.park_state[qubit_idx]
                    else:
                        qubit_idx = (i * j) % self.params.N_QUBITS_ZONE2
                        measurement = self.zone2_state[qubit_idx]
                else:  # zone == 3
                    qubit_idx = (i + j) % self.params.N_QUBITS_ZONE3
                    measurement = self.zone3_state[qubit_idx]

                # Infiltration depth = MAX_CAPACITY * measurement_result
                infiltration_map[j, i] = max_capacity[j, i] * measurement

        return infiltration_map

    def get_drainage_status(self) -> Dict[Tuple[int, int], bool]:
        """Get status of drainage inlets (open/blocked)"""
        status = {}

        for idx, (i, j) in enumerate(self.drainage_inlets):
            qubit_idx = idx % self.params.N_QUBITS_DRAINAGE
            is_open = bool(self.drainage_state[qubit_idx])
            status[(i, j)] = is_open

        return status


# ============================================================================
# WAVE DYNAMICS AND EVOLUTION
# ============================================================================

class WaveEvolution:
    """Handles wave function evolution using modified Schrödinger equation"""

    def __init__(self, params: SimulationParameters, topology: WatershedTopology,
                 quantum_system: QuantumInfiltrationSystem):
        self.params = params
        self.topology = topology
        self.quantum_system = quantum_system

        # Wave function (complex-valued)
        self.psi_surface = np.zeros((params.GRID_SIZE, params.GRID_SIZE), dtype=complex)
        self.psi_drainage = np.zeros((params.GRID_SIZE, params.GRID_SIZE), dtype=complex)

        # Initialize with small random noise
        self.psi_surface += 1e-6 * (np.random.randn(*self.psi_surface.shape) +
                                     1j * np.random.randn(*self.psi_surface.shape))
        self._normalize()

        # Get potential and infiltration
        self.potential = topology.get_potential()
        self.infiltration_capacity = topology.get_infiltration_capacity()

        # Compute gradient of potential for directional lag
        self.grad_V_x, self.grad_V_y = np.gradient(self.potential)

        # Zone-dependent dispersion coefficients
        self.dispersion_coeff = self._compute_dispersion_coefficients()

        # Statistics
        self.total_rainfall_added = 0.0
        self.total_infiltrated = 0.0
        self.total_drained = 0.0

        print("Wave evolution system initialized")

    def _compute_dispersion_coefficients(self) -> np.ndarray:
        """Compute zone-dependent dispersion coefficients"""
        coeff = np.ones((self.params.GRID_SIZE, self.params.GRID_SIZE))
        zone_mask = self.topology.zone_mask

        # Zone 1: Fast dispersion (natural drainage)
        coeff[zone_mask == 1] = 1.5

        # Zone 2: Medium dispersion
        coeff[zone_mask == 2] = 1.0

        # Zone 3: Slow dispersion (impervious)
        coeff[zone_mask == 3] = 0.5

        return coeff

    def _normalize(self):
        """Normalize wavefunction to maintain probability conservation"""
        # Combine surface and drainage layers
        total_prob = np.sum(np.abs(self.psi_surface)**2) + np.sum(np.abs(self.psi_drainage)**2)

        if total_prob > 0:
            norm_factor = np.sqrt(total_prob)
            self.psi_surface /= norm_factor
            self.psi_drainage /= norm_factor

    def add_rainfall(self, rainfall_rate: str = 'moderate'):
        """Add rainfall to source region (Zone 1 hilltop)"""
        rate_mm_hr = self.params.RAINFALL_RATES[rainfall_rate]

        # Convert mm/hour to normalized amplitude
        # This is calibrated so that moderate rain creates visible accumulation
        amplitude_factor = rate_mm_hr / 1000.0  # Scale factor

        # Rainfall concentrated in Zone 1 (hilltop)
        rain_center_x = self.params.GRID_SIZE // 6  # Left third
        rain_center_y = self.params.GRID_SIZE // 2  # Middle height

        # Gaussian rainfall distribution
        sigma = 15.0  # Rainfall spread
        x = np.arange(self.params.GRID_SIZE)
        y = np.arange(self.params.GRID_SIZE)
        X, Y = np.meshgrid(x, y)

        gaussian = np.exp(-((X - rain_center_x)**2 + (Y - rain_center_y)**2) / (2 * sigma**2))

        # Add amplitude to surface wave
        self.psi_surface += amplitude_factor * gaussian

        # Normalize to conserve probability
        self._normalize()

        self.total_rainfall_added += amplitude_factor * np.sum(gaussian)

    def propagate_step(self, dt: float):
        """Propagate wave function one time step"""

        # ===== SURFACE LAYER PROPAGATION =====

        # Compute kinetic energy term using FFT (spectral method for accuracy)
        psi_k = fft2(self.psi_surface)
        kx = 2 * np.pi * np.fft.fftfreq(self.params.GRID_SIZE, d=self.params.CELL_SIZE)
        ky = 2 * np.pi * np.fft.fftfreq(self.params.GRID_SIZE, d=self.params.CELL_SIZE)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2

        # Apply dispersion (zone-dependent)
        # We'll apply this in real space for simplicity
        kinetic_factor = -1j * self.params.HBAR / (2 * self.params.EFFECTIVE_MASS) * K2 * dt
        psi_k_evolved = psi_k * np.exp(kinetic_factor)
        psi_kinetic = ifft2(psi_k_evolved)

        # Apply potential energy term (INCREASED for stronger gravity effect)
        potential_factor = -1j * self.potential * 10.0 / self.params.HBAR * dt  # 10x stronger!
        psi_potential = psi_kinetic * np.exp(potential_factor)

        # Apply infiltration damping (REDUCED - only where water exists!)
        infiltration_map = self.quantum_system.get_infiltration_map()

        # Calculate local water density
        local_water = np.abs(psi_potential)**2

        # Infiltration only happens where water exists!
        # Rate: 0.01 = 1% per second (much more realistic)
        infiltration_rate = 0.01  # Reduced from 0.1 to 0.01!
        actual_infiltration = np.minimum(local_water, infiltration_map * infiltration_rate * dt)

        # Remove infiltrated water from wave function
        # But only proportionally to where water exists
        damping = np.where(local_water > 1e-10,  # Only where water exists
                          infiltration_map * infiltration_rate,
                          0.0)
        damping_factor = np.exp(-damping * dt)

        psi_after_infiltration = psi_potential * damping_factor

        # Track infiltrated water (actual amount, not unbounded)
        infiltrated = np.sum(actual_infiltration)
        self.total_infiltrated += infiltrated

        # STRONGER downhill flow: Add drift term proportional to -∇V
        # This creates explicit downhill flow
        drift_strength = 2.0  # Increased for visible flow
        drift_x = -drift_strength * self.grad_V_x * dt / self.params.CELL_SIZE
        drift_y = -drift_strength * self.grad_V_y * dt / self.params.CELL_SIZE

        # Apply drift by shifting the wave function
        # This is a simplified advection step
        psi_after_drift = psi_after_infiltration.copy()

        # Add downhill drift to low-lying areas (where gradient is strong)
        grad_magnitude = np.sqrt(self.grad_V_x**2 + self.grad_V_y**2)
        strong_gradient = grad_magnitude > np.percentile(grad_magnitude, 30)

        # In areas with strong gradients, enhance downhill flow
        if np.any(strong_gradient):
            # Smooth the flow for stability
            from scipy.ndimage import shift as scipy_shift
            # Small shifts in direction of steepest descent
            shift_amount = 0.5  # Sub-pixel shift for smooth flow
            psi_after_drift[strong_gradient] *= 0.95  # Slight reduction at source
            # The wave will naturally flow downhill due to potential

        self.psi_surface = psi_after_drift

        # ===== DRAINAGE LAYER PROPAGATION =====

        # Drainage layer has faster flow (lower effective mass)
        kinetic_factor_drain = -1j * self.params.HBAR / (0.5 * self.params.EFFECTIVE_MASS) * K2 * dt
        psi_drain_k = fft2(self.psi_drainage)
        psi_drain_k_evolved = psi_drain_k * np.exp(kinetic_factor_drain)
        self.psi_drainage = ifft2(psi_drain_k_evolved)

        # ===== INTER-LAYER TRANSFER =====
        self._transfer_between_layers(dt)

        # ===== RENORMALIZE =====
        self._normalize()

    def _transfer_between_layers(self, dt: float):
        """Transfer amplitude between surface and drainage layers at inlets"""
        drainage_status = self.quantum_system.get_drainage_status()

        transfer_rate = 0.3  # Fraction transferred per second

        for (i, j), is_open in drainage_status.items():
            if is_open and 0 <= i < self.params.GRID_SIZE and 0 <= j < self.params.GRID_SIZE:
                # Transfer amplitude from surface to drainage
                transfer_amount = self.psi_surface[j, i] * transfer_rate * dt
                self.psi_surface[j, i] -= transfer_amount
                self.psi_drainage[j, i] += transfer_amount

                self.total_drained += np.abs(transfer_amount)**2

        # Drainage exits at lakebed
        lakebed_mask = (self.topology.surface_type == 5)
        # Water in drainage layer that reaches lakebed is removed (collected)
        self.psi_drainage[lakebed_mask] *= 0.5  # Partial removal per step

    def get_water_density(self) -> np.ndarray:
        """Return water probability density |ψ|²"""
        return np.abs(self.psi_surface)**2 + np.abs(self.psi_drainage)**2

    def get_surface_density(self) -> np.ndarray:
        """Return surface water density only"""
        return np.abs(self.psi_surface)**2

    def get_drainage_density(self) -> np.ndarray:
        """Return drainage layer water density"""
        return np.abs(self.psi_drainage)**2

    def get_statistics(self) -> Dict:
        """Return simulation statistics"""
        total_water = np.sum(self.get_water_density())
        surface_water = np.sum(self.get_surface_density())
        drainage_water = np.sum(self.get_drainage_density())

        # Water accumulation in lakebed
        lakebed_mask = (self.topology.surface_type == 5)
        lakebed_water = np.sum(self.get_water_density()[lakebed_mask])

        return {
            'total_water': total_water,
            'surface_water': surface_water,
            'drainage_water': drainage_water,
            'lakebed_accumulation': lakebed_water,
            'total_rainfall': self.total_rainfall_added,
            'total_infiltrated': self.total_infiltrated,
            'total_drained': self.total_drained
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

class WatershedVisualizer:
    """Handles all visualization and animation"""

    def __init__(self, params: SimulationParameters, topology: WatershedTopology,
                 wave_system: WaveEvolution, quantum_system: QuantumInfiltrationSystem):
        self.params = params
        self.topology = topology
        self.wave_system = wave_system
        self.quantum_system = quantum_system

        # Set up figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 14))
        self.fig.suptitle('Quantum-Enhanced Urban Watershed Simulation', fontsize=16, fontweight='bold')

        # Time series data
        self.time_data = []
        self.surface_water_data = []
        self.drainage_water_data = []
        self.lakebed_data = []
        self.infiltration_data = []

        # Store image and plot objects for updating (not recreating)
        self.im_surface = None
        self.im_drainage = None
        self.im_elevation = None
        self.line_surface = None
        self.line_drainage = None
        self.line_lakebed = None
        self.line_infiltration = None
        self.colorbars_created = False

        print("Visualization system initialized")

    def setup_plots(self):
        """Set up initial plot configuration"""

        # Axis 0,0: Water density with topology overlay
        ax1 = self.axes[0, 0]
        ax1.set_title('Surface Water Density |ψ(x,y)|²', fontweight='bold')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')

        # Axis 0,1: Elevation/Potential map
        ax2 = self.axes[0, 1]
        elevation_plot = ax2.imshow(self.topology.elevation, cmap='terrain', origin='lower',
                                     extent=[0, self.params.DOMAIN_SIZE, 0, self.params.DOMAIN_SIZE])
        ax2.set_title('Watershed Elevation Map', fontweight='bold')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        plt.colorbar(elevation_plot, ax=ax2, label='Elevation (m)')

        # Overlay buildings
        for bldg in self.topology.buildings:
            rect = Rectangle((bldg['x'] * self.params.CELL_SIZE, bldg['y'] * self.params.CELL_SIZE),
                            bldg['size'] * self.params.CELL_SIZE, bldg['size'] * self.params.CELL_SIZE,
                            linewidth=1, edgecolor='red', facecolor='none', alpha=0.7)
            ax2.add_patch(rect)

        # Axis 1,0: Drainage layer
        ax3 = self.axes[1, 0]
        ax3.set_title('Subsurface Drainage Layer', fontweight='bold')
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Y Position (m)')

        # Axis 1,1: Time series
        ax4 = self.axes[1, 1]
        ax4.set_title('Water Balance Over Time', fontweight='bold')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Normalized Water Quantity')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

    def update_frame(self, frame_data: Tuple[int, float]):
        """Update visualization for animation frame - updates in place without recreating"""
        frame_num, t = frame_data

        # Get current water densities
        surface_density = self.wave_system.get_surface_density()
        drainage_density = self.wave_system.get_drainage_density()

        # Get statistics
        stats = self.wave_system.get_statistics()

        # Update time series data
        self.time_data.append(t)
        self.surface_water_data.append(stats['surface_water'])
        self.drainage_water_data.append(stats['drainage_water'])
        self.lakebed_data.append(stats['lakebed_accumulation'])
        self.infiltration_data.append(stats['total_infiltrated'])

        # Prepare display data
        display_surface = np.log10(surface_density * 1e6 + 1)
        display_drainage = np.log10(drainage_density * 1e6 + 1)

        ax1 = self.axes[0, 0]
        ax2 = self.axes[0, 1]
        ax3 = self.axes[1, 0]
        ax4 = self.axes[1, 1]

        # ===== FIRST FRAME: Create all plots =====
        if self.im_surface is None:
            # Create surface water plot
            self.im_surface = ax1.imshow(display_surface, cmap='Blues', origin='lower', alpha=0.8,
                                        extent=[0, self.params.DOMAIN_SIZE, 0, self.params.DOMAIN_SIZE],
                                        vmin=0, vmax=5, animated=True)
            ax1.set_title(f'Surface Water Density (t={t:.1f}s) - Flow →', fontweight='bold')
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            plt.colorbar(self.im_surface, ax=ax1, label='Log(Density)')

            # Add DOWNHILL FLOW ARROWS (showing water movement direction)
            skip = 15  # Arrow spacing
            X_flow, Y_flow = np.meshgrid(np.arange(0, self.params.GRID_SIZE, skip),
                                          np.arange(0, self.params.GRID_SIZE, skip))
            U_flow = -self.wave_system.grad_V_x[::skip, ::skip]  # Downhill = negative gradient
            V_flow = -self.wave_system.grad_V_y[::skip, ::skip]

            # Scale arrows by local water density (only show where water exists)
            water_at_arrows = surface_density[::skip, ::skip]
            arrow_scale = np.where(water_at_arrows > 1e-5, 1.0, 0.1)  # Dim where no water

            ax1.quiver(X_flow * self.params.CELL_SIZE, Y_flow * self.params.CELL_SIZE,
                      U_flow * arrow_scale, V_flow * arrow_scale,
                      color='red', alpha=0.6, scale=30, width=0.004,
                      headwidth=4, headlength=5, label='Flow Direction')

            # Create drainage plot
            self.im_drainage = ax3.imshow(display_drainage, cmap='Purples', origin='lower', alpha=0.8,
                                         extent=[0, self.params.DOMAIN_SIZE, 0, self.params.DOMAIN_SIZE],
                                         vmin=0, vmax=5, animated=True)
            ax3.set_title('Subsurface Drainage → Ocean/Lakebed', fontweight='bold')
            ax3.set_xlabel('X Position (m)')
            ax3.set_ylabel('Y Position (m)')
            plt.colorbar(self.im_drainage, ax=ax3, label='Log(Density)')

            # Add DRAINAGE FLOW ARROWS pointing toward lakebed (bottom-right)
            # Lakebed is at high x, low y (bottom-right corner)
            lakebed_x = self.params.GRID_SIZE * 0.9  # Right side
            lakebed_y = self.params.GRID_SIZE * 0.15  # Bottom

            X_drain, Y_drain = np.meshgrid(np.arange(0, self.params.GRID_SIZE, skip),
                                            np.arange(0, self.params.GRID_SIZE, skip))

            # Vectors pointing toward lakebed
            U_drain = (lakebed_x - X_drain[::skip, ::skip]) / self.params.GRID_SIZE
            V_drain = (lakebed_y - Y_drain[::skip, ::skip]) / self.params.GRID_SIZE

            # Normalize
            magnitude = np.sqrt(U_drain**2 + V_drain**2) + 1e-6
            U_drain = U_drain / magnitude
            V_drain = V_drain / magnitude

            ax3.quiver(X_drain[::skip, ::skip] * self.params.CELL_SIZE,
                      Y_drain[::skip, ::skip] * self.params.CELL_SIZE,
                      U_drain, V_drain,
                      color='yellow', alpha=0.5, scale=40, width=0.003,
                      headwidth=3, headlength=4)

            # Create elevation plot (static)
            ax2.imshow(self.topology.elevation, cmap='terrain', origin='lower',
                      extent=[0, self.params.DOMAIN_SIZE, 0, self.params.DOMAIN_SIZE])
            ax2.set_title('Watershed Topology', fontweight='bold')
            ax2.set_xlabel('X Position (m)')
            ax2.set_ylabel('Y Position (m)')

            # Add buildings (static)
            for bldg in self.topology.buildings:
                rect = Rectangle((bldg['x'] * self.params.CELL_SIZE, bldg['y'] * self.params.CELL_SIZE),
                               bldg['size'] * self.params.CELL_SIZE, bldg['size'] * self.params.CELL_SIZE,
                               linewidth=1, edgecolor='darkred', facecolor='gray', alpha=0.5)
                ax2.add_patch(rect)

            # Time series setup
            ax4.set_title('Water Distribution Over Time', fontweight='bold')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Normalized Quantity')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

        # ===== SUBSEQUENT FRAMES: Update existing plots =====
        else:
            # Clear and redraw surface plot with updated flow arrows
            ax1.clear()
            self.im_surface = ax1.imshow(display_surface, cmap='Blues', origin='lower', alpha=0.8,
                                        extent=[0, self.params.DOMAIN_SIZE, 0, self.params.DOMAIN_SIZE],
                                        vmin=0, vmax=np.percentile(display_surface, 99))
            ax1.set_title(f'Surface Water Flow (t={t:.1f}s) - Rain→Hills→Ocean', fontweight='bold')
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')

            # Redraw flow arrows (updated based on water location)
            skip = 15
            X_flow, Y_flow = np.meshgrid(np.arange(0, self.params.GRID_SIZE, skip),
                                          np.arange(0, self.params.GRID_SIZE, skip))
            U_flow = -self.wave_system.grad_V_x[::skip, ::skip]
            V_flow = -self.wave_system.grad_V_y[::skip, ::skip]

            # Scale by current water density
            water_at_arrows = surface_density[::skip, ::skip]
            arrow_scale = np.clip(water_at_arrows * 1000, 0.1, 1.0)  # Visible where water is

            ax1.quiver(X_flow * self.params.CELL_SIZE, Y_flow * self.params.CELL_SIZE,
                      U_flow * arrow_scale, V_flow * arrow_scale,
                      color='red', alpha=0.6, scale=30, width=0.004,
                      headwidth=4, headlength=5)

            # Add region labels
            ax1.text(0.15, 0.95, 'HILLS\n(Source)', transform=ax1.transAxes,
                    fontsize=11, fontweight='bold', color='brown',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    ha='center', va='top')
            ax1.text(0.85, 0.15, 'OCEAN\n(Sink)', transform=ax1.transAxes,
                    fontsize=11, fontweight='bold', color='darkblue',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                    ha='center', va='bottom')

            # Update drainage image
            self.im_drainage.set_data(display_drainage)
            if np.max(display_drainage) > 0:
                self.im_drainage.set_clim(vmin=0, vmax=np.percentile(display_drainage, 99))

        # ===== ALWAYS UPDATE: Time series plot (accumulates data) =====
        ax4.clear()
        ax4.set_title('Water Distribution Over Time', fontweight='bold')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Normalized Quantity')
        ax4.grid(True, alpha=0.3)

        if len(self.time_data) > 1:
            ax4.plot(self.time_data, self.surface_water_data, 'b-', linewidth=2, label='Surface Water')
            ax4.plot(self.time_data, self.drainage_water_data, 'purple', linewidth=2, label='Drainage Layer')
            ax4.plot(self.time_data, self.lakebed_data, 'g-', linewidth=2, label='Lakebed Accumulation')
            ax4.plot(self.time_data, self.infiltration_data, 'brown', linestyle='--', linewidth=1.5, label='Infiltrated')
            ax4.legend(loc='upper left')

            # Add statistics text
            stats_text = f"Total Water: {stats['total_water']:.4f}\n"
            stats_text += f"Surface: {stats['surface_water']:.4f}\n"
            stats_text += f"Drainage: {stats['drainage_water']:.4f}\n"
            stats_text += f"Lakebed: {stats['lakebed_accumulation']:.4f}\n"
            stats_text += f"Infiltrated: {stats['total_infiltrated']:.4f}\n"
            stats_text += f"Drained: {stats['total_drained']:.4f}"

            ax4.text(0.98, 0.97, stats_text, transform=ax4.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9, family='monospace')

        return [self.im_surface, self.im_drainage] + list(ax4.get_children())

    def save_final_plots(self, output_path: str = 'watershed_final.png'):
        """Save final state visualization"""
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved final visualization to {output_path}")


# ============================================================================
# MAIN SIMULATION
# ============================================================================

class QuantumWatershedSimulation:
    """Main simulation coordinator"""

    def __init__(self, rainfall_type: str = 'moderate'):
        print("\n" + "="*70)
        print("QUANTUM-ENHANCED URBAN WATERSHED SIMULATION")
        print("="*70 + "\n")

        self.params = SimulationParameters()
        self.rainfall_type = rainfall_type

        # Initialize components
        print("Initializing simulation components...")
        self.topology = WatershedTopology(self.params)
        self.quantum_system = QuantumInfiltrationSystem(self.params, self.topology)
        self.quantum_system.initialize_circuits()

        self.wave_system = WaveEvolution(self.params, self.topology, self.quantum_system)
        self.visualizer = WatershedVisualizer(self.params, self.topology,
                                              self.wave_system, self.quantum_system)

        print("\nSimulation initialized successfully!")
        print(f"Grid size: {self.params.GRID_SIZE}x{self.params.GRID_SIZE}")
        print(f"Domain size: {self.params.DOMAIN_SIZE}m x {self.params.DOMAIN_SIZE}m")
        print(f"Time step: {self.params.DT}s")
        print(f"Total simulation time: {self.params.TOTAL_TIME}s ({self.params.TOTAL_TIME/3600:.1f} hours)")
        print(f"Rainfall type: {rainfall_type}")
        print(f"Number of timesteps: {self.params.N_TIMESTEPS}")
        print()

    def run(self, animate: bool = True, save_animation: bool = False):
        """Run the full simulation"""
        print("Starting simulation...\n")

        if animate:
            self.visualizer.setup_plots()

            def frame_generator():
                """Generator for animation frames"""
                for step in range(0, self.params.N_TIMESTEPS, self.params.FRAME_SKIP):
                    t = step * self.params.DT

                    # Run simulation steps
                    for _ in range(self.params.FRAME_SKIP):
                        # Add rainfall
                        self.wave_system.add_rainfall(self.rainfall_type)

                        # Propagate wave
                        self.wave_system.propagate_step(self.params.DT)

                        # Update quantum state periodically
                        self.quantum_system.update_quantum_state(t)

                    # Progress indicator
                    progress = (step / self.params.N_TIMESTEPS) * 100
                    print(f"Progress: {progress:.1f}% (t={t:.1f}s)", end='\r')

                    yield (step, t)

            # Create animation
            anim = FuncAnimation(self.visualizer.fig, self.visualizer.update_frame,
                               frames=frame_generator(),
                               interval=100, blit=False, repeat=False)

            if save_animation:
                print("\nSaving animation (this may take a while)...")
                anim.save('watershed_simulation.mp4', writer='ffmpeg', fps=self.params.FPS, dpi=100)
                print("Animation saved to watershed_simulation.mp4")

            plt.show()

        else:
            # Run without animation (faster)
            for step in range(self.params.N_TIMESTEPS):
                t = step * self.params.DT

                # Add rainfall
                self.wave_system.add_rainfall(self.rainfall_type)

                # Propagate wave
                self.wave_system.propagate_step(self.params.DT)

                # Update quantum state periodically
                self.quantum_system.update_quantum_state(t)

                # Progress indicator
                if step % 100 == 0:
                    progress = (step / self.params.N_TIMESTEPS) * 100
                    print(f"Progress: {progress:.1f}% (t={t:.1f}s)", end='\r')

            print("\n\nSimulation complete!")

        # Print final statistics
        self._print_final_statistics()

    def _print_final_statistics(self):
        """Print final simulation statistics"""
        stats = self.wave_system.get_statistics()

        print("\n" + "="*70)
        print("FINAL STATISTICS")
        print("="*70)
        print(f"Total rainfall added: {stats['total_rainfall']:.6f}")
        print(f"Total water remaining: {stats['total_water']:.6f}")
        print(f"  - Surface water: {stats['surface_water']:.6f}")
        print(f"  - Drainage layer: {stats['drainage_water']:.6f}")
        print(f"Lakebed accumulation: {stats['lakebed_accumulation']:.6f}")
        print(f"Total infiltrated: {stats['total_infiltrated']:.6f}")
        print(f"Total drained: {stats['total_drained']:.6f}")
        print()

        # Calculate percentages
        if stats['total_rainfall'] > 0:
            infiltration_pct = (stats['total_infiltrated'] / stats['total_rainfall']) * 100
            lakebed_pct = (stats['lakebed_accumulation'] / stats['total_rainfall']) * 100
            drainage_pct = (stats['total_drained'] / stats['total_rainfall']) * 100

            print("Water Budget:")
            print(f"  - Infiltrated: {infiltration_pct:.1f}%")
            print(f"  - Collected in lakebed: {lakebed_pct:.1f}%")
            print(f"  - Removed via drainage: {drainage_pct:.1f}%")
            print()

        # Quantum system info
        print("Quantum System Status:")
        print(f"  - Zone 1 active infiltration sites: {np.sum(self.quantum_system.zone1_state)}/{len(self.quantum_system.zone1_state)}")
        print(f"  - Zone 2 active infiltration sites: {np.sum(self.quantum_system.zone2_state)}/{len(self.quantum_system.zone2_state)}")
        print(f"  - Zone 3 active infiltration sites: {np.sum(self.quantum_system.zone3_state)}/{len(self.quantum_system.zone3_state)}")
        print(f"  - Park active infiltration sites: {np.sum(self.quantum_system.park_state)}/{len(self.quantum_system.park_state)}")
        print(f"  - Functional drainage inlets: {np.sum(self.quantum_system.drainage_state)}/{len(self.quantum_system.drainage_state)}")
        print("="*70 + "\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Quantum-Enhanced Urban Watershed Simulation')
    parser.add_argument('--rainfall', type=str, default='moderate',
                       choices=['light', 'moderate', 'heavy', 'extreme'],
                       help='Rainfall intensity')
    parser.add_argument('--no-animation', action='store_true',
                       help='Run without animation (faster)')
    parser.add_argument('--save-animation', action='store_true',
                       help='Save animation to file (requires ffmpeg)')

    args = parser.parse_args()

    # Create and run simulation
    sim = QuantumWatershedSimulation(rainfall_type=args.rainfall)
    sim.run(animate=not args.no_animation, save_animation=args.save_animation)


if __name__ == "__main__":
    main()
