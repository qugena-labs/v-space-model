# Quantum-Enhanced Urban Watershed Simulation

A comprehensive 2D+1 spatial simulation integrating quantum mechanics principles with hydrological processes to model rainfall and water flow in an urban watershed environment.

## Overview

This simulation combines:
- **Quantum Computing**: Qiskit-based quantum circuits control infiltration dynamics and drainage system behavior
- **Hydrological Physics**: Modified 2D Schrödinger equation models water flow as a probability wave
- **Urban Topology**: Realistic 3-zone watershed with buildings, roads, parks, and drainage infrastructure
- **Real-time Visualization**: Multi-panel animated display of water dynamics

## Key Features

### Physical Modeling
- **Wave Function Dynamics**: Water represented as quantum wavefunction ψ(x,y,t) with |ψ|² as probability density
- **Gravitational Flow**: Elevation-based potential energy guides water downhill
- **Directional Lag**: Uphill propagation slower than downhill (realistic physics)
- **Probability Conservation**: Strict normalization ensures mass balance

### Quantum Subsystems
- **Three-Zone Quantum Registers**: Separate quantum circuits for hill, suburban, and urban zones
- **Infiltration Control**: Qubit measurements determine microwell depth at each location
- **Drainage Network**: Quantum-controlled inlet states (open/blocked)
- **Decoherence Models**: Zone-specific noise simulating environmental effects
- **Park Enhancement**: Dedicated sub-register for green infrastructure

### Urban Infrastructure
- **Three Distinct Zones**:
  1. **LEFT (Hills)**: Residential area on steep slope, high elevation (50-100m)
  2. **MIDDLE (Suburban)**: Mixed-use with central park, gentle slope (20-50m)
  3. **RIGHT (Urban/Lakebed)**: High-rise buildings and low-elevation collection basin (0-20m)

- **Building Types**:
  - Residential houses (5x5 to 10x10 cells)
  - Skyscrapers (8x8 to 15x15 cells, 30-80m height)
  - Park structure (3x3 cell "tree")

- **Road Network**:
  - Curved roads using spline interpolation
  - Sidewalks elevated 0.3m above roads
  - Drainage inlets every ~20 cells

- **Two-Layer System**:
  - Surface layer (z=0): Visible water flow
  - Subsurface drainage (z=-1): Storm sewer network

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) ffmpeg for saving animations

### Setup

```bash
# Clone or download the repository
cd quantumadvantage_modeling

# Install dependencies
pip install -r requirements_watershed.txt

# Verify Qiskit installation
python -c "import qiskit; print(qiskit.__version__)"
```

### Installing FFmpeg (Optional, for animation export)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

## Usage

### Basic Run

```bash
python quantum_watershed_simulation.py
```

This will run a simulation with:
- Moderate rainfall intensity
- Real-time animation display
- 1 hour simulated time (~5-10 minutes runtime)

### Command Line Options

```bash
# Light rainfall
python quantum_watershed_simulation.py --rainfall light

# Heavy rainfall
python quantum_watershed_simulation.py --rainfall heavy

# Extreme storm event
python quantum_watershed_simulation.py --rainfall extreme

# Run without animation (much faster)
python quantum_watershed_simulation.py --no-animation

# Save animation to file
python quantum_watershed_simulation.py --save-animation
```

### Available Rainfall Types

| Type | Rate (mm/hour) | Description |
|------|----------------|-------------|
| `light` | 5 | Light drizzle |
| `moderate` | 15 | Steady rain |
| `heavy` | 30 | Heavy rainstorm |
| `extreme` | 80 | Extreme precipitation event |

## Output and Visualization

### Real-time Display (4 Panels)

**Panel 1 (Top-Left): Surface Water Density**
- Heatmap of |ψ(x,y)|² (water probability)
- Topology overlay (roads, buildings, parks)
- Drainage inlet markers (green=open, red=blocked)

**Panel 2 (Top-Right): Watershed Topology**
- Elevation map with terrain colors
- Building footprints (red outlines)
- Flow direction vectors (blue arrows)

**Panel 3 (Bottom-Left): Subsurface Drainage**
- Water in underground storm sewer system
- Lakebed collection area (green highlight)
- Shows inter-layer transfer effectiveness

**Panel 4 (Bottom-Right): Time Series**
- Surface water quantity over time
- Drainage layer accumulation
- Lakebed collection
- Total infiltration
- Statistical summary box

### Saved Output

- **Final visualization**: `watershed_final.png`
- **Animation video**: `watershed_simulation.mp4` (if `--save-animation` used)

## Scientific Details

### Modified Schrödinger Equation

The simulation solves:

```
iℏ ∂ψ/∂t = -ℏ²/(2m) ∇²ψ + V(x,y)ψ - iΓ(x,y)ψ
```

Where:
- `ψ(x,y,t)`: Complex-valued wavefunction (water probability amplitude)
- `V(x,y)`: Gravitational potential (elevation + buildings)
- `Γ(x,y)`: Quantum-controlled infiltration damping
- `m`: Effective mass (controls flow speed)

### Quantum Circuit Structure

**Infiltration Circuit:**
1. Initialize qubits in |0⟩ state
2. Apply Hadamard gates → superposition
3. Controlled rotations based on surface properties
4. Apply depolarizing noise (decoherence)
5. Measure → classical bitstring
6. Map to infiltration depth

**Drainage Circuit:**
1. Initialize with bias toward |1⟩ (functional)
2. Entangle adjacent inlets (interconnected system)
3. Measure → inlet status

### Normalization and Mass Conservation

After each rainfall addition:
```python
total_probability = ∫∫|ψ|² dx dy
ψ_normalized = ψ / sqrt(total_probability)
```

This ensures:
- Total probability = 1 at all times
- Mass conservation throughout simulation
- Physically meaningful water balance

### Zone-Specific Parameters

| Parameter | Zone 1 (Hills) | Zone 2 (Suburban) | Zone 3 (Urban) |
|-----------|----------------|-------------------|----------------|
| Qubits | 7 | 10 | 15 |
| Decoherence | 0.3 s⁻¹ (fast) | 0.15 s⁻¹ (medium) | 0.05 s⁻¹ (slow) |
| Dispersion | 1.5× (fast) | 1.0× (medium) | 0.5× (slow) |
| Max Infiltration | 20 cm | 10-40 cm | 5 cm |

## Performance and Scalability

### Default Configuration
- Grid: 150×150 cells
- Domain: 750m × 750m
- Time step: 2 seconds
- Simulation duration: 1 hour (1800 steps)
- Runtime: ~5-10 minutes with animation

### Optimization Tips

1. **Reduce grid size** for faster testing:
   ```python
   SimulationParameters.GRID_SIZE = 100  # Instead of 150
   ```

2. **Increase time step** (less accurate but faster):
   ```python
   SimulationParameters.DT = 5.0  # Instead of 2.0
   ```

3. **Reduce visualization frames**:
   ```python
   SimulationParameters.FRAME_SKIP = 20  # Instead of calculated value
   ```

4. **Run without animation**:
   ```bash
   python quantum_watershed_simulation.py --no-animation
   ```

## Understanding the Results

### Water Budget Analysis

The simulation tracks:
- **Total Rainfall Added**: Cumulative probability amplitude injected
- **Surface Water**: Water visible on surface layer
- **Drainage Layer**: Water in subsurface system
- **Lakebed Accumulation**: Water collected in terminal basin
- **Infiltrated**: Water lost to soil absorption
- **Drained**: Water removed via drainage system

### Expected Behavior

**Light Rain:**
- Gradual accumulation in lakebed
- Most water infiltrates
- Little surface runoff

**Moderate Rain:**
- Visible flow from hills to lakebed
- Balanced infiltration and runoff
- Drainage system engaged

**Heavy/Extreme Rain:**
- Rapid surface accumulation
- Overwhelmed infiltration capacity
- Significant drainage layer activation
- Potential flooding in low-lying areas

### Quantum vs. Classical

The quantum approach provides:
- **Stochastic Infiltration**: Natural variability in soil conditions
- **Correlated Failures**: Drainage inlets can fail together (entanglement)
- **Emergent Behavior**: Complex patterns from simple quantum rules
- **Physical Realism**: Uncertainty inherent in measurements

## Extending the Simulation

### Adding Custom Scenarios

Modify `quantum_watershed_simulation.py`:

**1. Storm Surge (Sudden Heavy Rain):**
```python
def add_storm_surge(self, t):
    if 500 < t < 700:  # 2-minute intense burst
        self.add_rainfall('extreme')
    else:
        self.add_rainfall('moderate')
```

**2. Drainage System Failure:**
```python
def induce_drainage_failure(self):
    # Set more qubits to |0⟩ (blocked)
    self.drainage_state[:] = 0
```

**3. Green Infrastructure Impact:**
```python
# Increase park infiltration
INFILTRATION_DEPTH_PARK = 0.80  # 80cm instead of 40cm
```

**4. Climate Change (Increased Intensity):**
```python
RAINFALL_RATES = {
    'moderate': 25.0,  # Was 15.0
    'heavy': 50.0,     # Was 30.0
    'extreme': 120.0   # Was 80.0
}
```

### Adding New Analysis

Flood risk mapping:
```python
def compute_flood_risk(self):
    """Identify regions where water exceeds threshold"""
    density = self.get_water_density()
    threshold = 0.01  # Adjust based on normalization
    flood_zones = density > threshold
    return flood_zones
```

## Troubleshooting

### Common Issues

**1. Qiskit Import Error**
```
ImportError: No module named 'qiskit'
```
Solution: `pip install qiskit qiskit-aer`

**2. Visualization Not Showing**
```
Backend error: TkAgg not available
```
Solution: Install tkinter or switch backend:
```python
matplotlib.use('Qt5Agg')  # Add before importing pyplot
```

**3. Animation Saving Fails**
```
ValueError: Cannot save animation - ffmpeg not found
```
Solution: Install ffmpeg system-wide (see Installation section)

**4. Simulation Runs Slowly**
- Reduce `GRID_SIZE` (line 76)
- Increase `DT` (line 79)
- Use `--no-animation` flag
- Reduce `TOTAL_TIME` (line 80)

**5. Quantum Backend Errors**
```
QiskitError: Unable to find simulator
```
Solution: Ensure qiskit-aer is installed: `pip install qiskit-aer`

### Fallback Classical Mode

If Qiskit is unavailable, the simulation automatically uses classical random sampling instead of quantum circuits. All features remain functional, but without true quantum behavior.

## Scientific Applications

This simulation is useful for:

1. **Urban Planning**: Evaluate drainage infrastructure capacity
2. **Flood Risk Assessment**: Identify vulnerable areas during extreme events
3. **Green Infrastructure Design**: Test park placement effectiveness
4. **Climate Adaptation**: Model future precipitation scenarios
5. **Quantum Algorithm Research**: Test quantum advantage in physical simulations
6. **Education**: Demonstrate quantum computing in real-world context

## Technical Architecture

```
quantum_watershed_simulation.py
├── SimulationParameters: Configuration constants
├── WatershedTopology: Spatial structure generation
│   ├── Elevation maps (3 zones)
│   ├── Building placement
│   ├── Road network (spline interpolation)
│   └── Park creation
├── QuantumInfiltrationSystem: Quantum subsystem
│   ├── Zone-specific quantum registers
│   ├── Circuit creation and execution
│   ├── Drainage inlet control
│   └── Measurement and state updates
├── WaveEvolution: Physical dynamics
│   ├── Schrödinger equation solver (FFT-based)
│   ├── Rainfall injection with renormalization
│   ├── Infiltration damping (quantum-controlled)
│   ├── Inter-layer transfer (drainage)
│   └── Statistics tracking
├── WatershedVisualizer: Rendering and animation
│   ├── Multi-panel layout
│   ├── Real-time density plots
│   ├── Time series graphs
│   └── Animation export
└── QuantumWatershedSimulation: Main coordinator
    ├── Component initialization
    ├── Simulation loop
    └── Output generation
```

## References and Further Reading

### Quantum Computing
- Qiskit Documentation: https://qiskit.org/documentation/
- Quantum Noise Models: https://qiskit.org/documentation/aer/tutorials/2_building_noise_models.html

### Hydrological Modeling
- Urban Watershed Dynamics
- Storm Water Management Systems
- Green Infrastructure Best Practices

### Mathematical Physics
- Schrödinger Equation in 2D
- Spectral Methods for PDEs
- Quantum-Classical Hybrid Systems

## License and Attribution

Created for quantum-classical hybrid systems research in hydrological modeling.

## Contact and Support

For questions, issues, or contributions:
- Open an issue in the repository
- Check documentation at https://docs.claude.com
- Review code comments for implementation details

---

**Version**: 1.0
**Last Updated**: 2025-10-28
**Tested With**: Python 3.9+, Qiskit 1.0+
