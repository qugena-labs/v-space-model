# V-Space Model for Urban Hydrological Systems

A quantum mechanics-based approach to modeling urban water drainage using potential energy landscapes (V-Space). This framework treats water flow as quantum wave packets evolving through spatially-varying potential fields, enabling physically realistic simulations of drainage capture, momentum conservation, and multi-scale urban hydrology.

## Table of Contents

- [Overview](#overview)
- [Theoretical Foundation](#theoretical-foundation)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Examples](#examples)
- [Parameters](#parameters)
- [Output](#output)

---

## Overview

The **V-Space Model** applies quantum mechanical principles to urban hydrological systems by:

1. **Potential Energy Landscapes**: Urban features (roads, buildings, drainage systems) are represented as potential fields V(x,y)
2. **Wave Packet Dynamics**: Water is modeled as quantum wave packets with momentum and energy
3. **Drainage Wells**: Drainage infrastructure creates deep potential wells that trap and capture water
4. **Momentum Conservation**: Forces, friction, and gravitational effects maintain physical realism

This approach naturally captures:
- Water accumulation in low-lying areas
- Drainage system capture efficiency
- Flow momentum and inertia
- Multi-layer drainage systems
- Long-duration rainfall events

---

## Theoretical Foundation

### Quantum Mechanics Approach

The system evolves according to the time-dependent Schrödinger equation:

```
iℏ ∂ψ/∂t = Ĥψ
```

Where the Hamiltonian Ĥ includes:
- **Kinetic energy**: Hopping between adjacent sites (∝ -∇²)
- **Potential energy**: V(x) representing urban landscape
- **Vertical coupling**: Connects surface and drainage layers

### Tight-Binding Hamiltonian

For a 1D system with N sites:

```
H[i,i] = V[i] + 2t        (on-site potential + kinetic)
H[i,i±1] = -t             (nearest-neighbor hopping)
```

Time evolution operator:
```
U(dt) = exp(-iH·dt/ℏ)
```

### Classical Force Dynamics

Surface water also follows classical momentum equations:

```
F = -∇V(x,y) - friction·v
v(t+dt) = v(t) + F·dt
```

This hybrid approach captures both quantum coherence effects and macroscopic flow.

---

## Installation

### Requirements

```bash
# Python 3.8+
pip install numpy matplotlib scipy qiskit qiskit-aer
```

### Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Visualization and animation
- **scipy**: Matrix exponentials and linear algebra
- **qiskit**: Quantum circuit construction
- **qiskit-aer**: Quantum statevector simulation

### Quick Start

```bash
git clone <repository>
cd quantumadvantage_modeling
pip install -r requirements.txt
```

---

## Project Structure

### Core Simulation Files

#### 1. `test_potential_sys.py`
**1D Quantum Drainage Well Simulation**

- **Purpose**: Fundamental 1D model of quantum wave packet falling into drainage well
- **Features**:
  - Single potential well with adjustable depth
  - Gaussian wave packet with initial momentum
  - Real-time animation of wave evolution
  - Energy conservation tracking

**Key Physics**:
```
Surface:  V = 0     ═══════╗              ╔═══════
                           ║  DRAINAGE   ║
Drainage: V = -5.0  ───────▼──────────────║
```

**Parameters**:
- `NUM_QUBITS`: System size (8 = 256 sites)
- `INITIAL_MOMENTUM`: Rightward velocity (0.45)
- `drainage_depth`: Well depth (-5.0 for strong capture)
- `TIME_STEPS`: Evolution duration (600)
- `DT`: Time step size (0.5)

---

#### 2. `two_layer_drainage_quantum.py`
**Two-Layer Quantum Drainage System**

- **Purpose**: Full 2D system with separate surface and drainage layers
- **Features**:
  - Independent top and bottom layers
  - Vertical coupling through drainage holes
  - Momentum transfer between layers
  - Dual-panel animation (surface + drainage)

**Key Physics**:
```
TOP LAYER:     [Surface water] --hole-->
BOTTOM LAYER:                    [Drainage system]
```

**Hamiltonian Structure**:
- Horizontal hopping within each layer
- Vertical coupling: `H[top_i, bottom_i] = -coupling`
- Adjustable coupling strength at hole locations

**Parameters**:
- `NUM_QUBITS`: 7 (128 sites per layer, 256 total)
- `SURFACE_HEIGHT`: 10.0 (barrier height)
- `DRAINAGE_HEIGHT`: 0.0 (low energy state)
- `vertical_coupling`: 0.5 (hole coupling strength)
- `HOLE_START/END`: 35-45% of domain

---

#### 3. `city_rainfall_simulation.py`
**Full Urban Rainfall Simulation with Momentum**

- **Purpose**: Complete city-scale simulation with buildings, roads, and drainage
- **Features**:
  - Time-dependent rainfall with multiple duration modes
  - Force-based momentum physics (F = -∇V - friction·v)
  - Crowned road surfaces for drainage
  - Separate velocity fields for surface and drainage water
  - Real-time statistics and visualization

**Key Physics**:
- **Gravity force**: `F = -gravity_strength * ∇V`
- **Friction**: `F_friction = -friction_coeff * v`
- **Momentum update**: `v(t+dt) = v(t) + (F_gravity + F_friction)·dt`
- **Drainage capture**: Fast capture rate for water over wells

**Rainfall Duration Modes**:
- `standard`: 50 steps (normal storm)
- `storm`: 150 steps (continuous storm)
- `day`: 300 steps (24-hour rain)
- `week`: 700 steps (7-day rain)
- `10day`: 1000 steps (10-day continuous)
- `month`: 1500 steps (30-day extreme event)

**Rainfall Intensities**:
- `light`: 0.02 depth per step
- `moderate`: 0.05 depth per step
- `heavy`: 0.10 depth per step
- `severe`: 0.20 depth per step

**Command Line Usage**:
```bash
python city_rainfall_simulation.py --rainfall moderate --duration 10day
python city_rainfall_simulation.py --rainfall severe --duration storm
python city_rainfall_simulation.py --rainfall heavy --duration month
```

---

#### 4. `static_city_potential.py`
**Urban Landscape Potential Generator**

- **Purpose**: Creates realistic urban potential energy landscapes
- **Features**:
  - Rectangular buildings with sloped edges
  - Crowned road surfaces (elevated center, depressed edges)
  - Drainage wells and infiltration systems
  - Permeable park areas

**City Layout**:
```
┌────────┐  ┌────────┐
│Building│  │Building│
│  V=50  │  │  V=50  │
└────────┘  └────────┘
    ║ Road ║
    ╚══╬══╝  (crowned)
       ║
   [Drainage]
    (V=-well)
```

**Road Crown Profile**:
- Center: Elevated by 1.5 units
- Edges: Depressed by -4.0 units (toward drainage)
- Profile: Parabolic (dist²)

**Drainage System**:
- Well depth: 0.15x surface wells (shallow capture)
- Spacing: Strategic placement near buildings
- Infiltration zones: Enhanced in park areas

---

#### 5. `simple_wave_drainage.py`
**Classical Wave Drainage Simulation (Optional)**

- **Purpose**: Classical wave equation approach for comparison
- **Features**:
  - 1D classical wave propagation
  - Simplified drainage interaction
  - Fast computation for testing

---

## Usage

### 1. Basic Quantum Drainage Simulation

**Run the 1D quantum drainage well**:
```bash
python test_potential_sys.py
```

**What it shows**:
- Wave packet starting on surface (V=0)
- Traveling rightward with momentum
- Encountering drainage well at 35-50%
- Getting trapped and oscillating in well
- Probability density animation

**Typical output**:
```
======================================================================
DRAINAGE QUANTUM WAVE SIMULATION
======================================================================
  Sites: 256
  Surface level: V = 0 (ground)
  Drainage well depth: V = -5.0 (DEEP POTENTIAL WELL)
  Well location: 35-50% (opening to drainage)
  Initial position: 38
  Initial momentum: 0.45
  Wave sits ON TOP of surface, FALLS and GETS TRAPPED in drainage well
======================================================================
```

---

### 2. Two-Layer Quantum System

**Run the two-layer simulation**:
```bash
python two_layer_drainage_quantum.py
```

**What it shows**:
- Two separate panels: TOP (surface) and BOTTOM (drainage)
- Wave starting on surface layer
- Probability transfer through hole
- Statistics showing distribution between layers

**Key observations**:
- Watch how probability flows from top to bottom
- Coupling strength determines transfer rate
- Momentum is conserved across layers

---

### 3. Full City Rainfall Simulation

**Run city-scale simulation**:
```bash
# Moderate 10-day rain
python city_rainfall_simulation.py --rainfall moderate --duration 10day

# Severe storm
python city_rainfall_simulation.py --rainfall severe --duration storm

# Long-term heavy rainfall (30 days)
python city_rainfall_simulation.py --rainfall heavy --duration month
```

**What it shows**:
- Real-time animation of water depth
- Buildings (red), roads (tan), parks (green), drainage (blue)
- Statistics panel showing:
  - Total water on surface
  - Total water in drainage
  - Capture efficiency
  - Infiltration rate

**Typical progression**:
1. Rain falls uniformly across city
2. Water flows toward drainage wells (momentum-driven)
3. Drainage captures water (fast rate)
4. Water continues past flat areas (momentum conservation)
5. System reaches steady state or overflow

---

## Examples

### Example 1: Testing Drainage Efficiency

Compare different drainage well depths:

```python
# Edit test_potential_sys.py
drainage_depth = -3.0   # Shallow well (weak capture)
drainage_depth = -5.0   # Medium well (good capture)
drainage_depth = -10.0  # Deep well (strong capture)
```

**Expected behavior**:
- Shallow: Wave partially reflects, partially trapped
- Medium: Most of wave captured with some reflection
- Deep: Complete capture, strong oscillations

---

### Example 2: Storm Duration Analysis

Test infrastructure under different rainfall durations:

```bash
# Short storm (standard)
python city_rainfall_simulation.py --rainfall heavy --duration standard

# Extended storm (10 days)
python city_rainfall_simulation.py --rainfall heavy --duration 10day

# Extreme event (30 days)
python city_rainfall_simulation.py --rainfall heavy --duration month
```

**Analysis questions**:
- When does drainage system saturate?
- Does surface water overflow?
- What is steady-state water depth?

---

### Example 3: Momentum vs. No Momentum

Compare with/without momentum physics:

```python
# In city_rainfall_simulation.py

# WITH momentum (current):
force_x = -self.gravity_strength * grad_x
self.velocity_x += force_x * self.dt

# WITHOUT momentum (set to zero):
self.velocity_x = 0
self.velocity_y = 0
```

**Expected difference**:
- With momentum: Water flows past flat areas, realistic behavior
- Without momentum: Water stops immediately on flat surfaces

---

## Parameters

### Quantum System Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_QUBITS` | 8 | System size (2^N sites) |
| `TIME_STEPS` | 600 | Number of evolution steps |
| `DT` | 0.5 | Time step size |
| `hopping_strength` | 1.0 | Kinetic energy term |
| `drainage_depth` | -5.0 | Well depth (negative = attractive) |

### City Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRID_SIZE` | 100 | Spatial resolution |
| `DT` | 0.1 | Time step |
| `gravity_strength` | 0.5 | Force magnitude |
| `friction_coeff` | 0.3 | Velocity damping |
| `drainage_capture_rate` | 0.3 | Water removal rate |
| `drainage_velocity_multiplier` | 1.5 | Faster drainage flow |

### Landscape Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `crown_height` | 1.5 | Road center elevation |
| `edge_depression` | -4.0 | Road edge slope |
| `well_depth_multiplier` | 0.15 | Shallow drainage wells |
| `building_height` | 50.0 | Building potential |

---

## Output

### Animation Display

All simulations produce real-time animations showing:

1. **Probability/Water Density**: Blue shading
2. **Potential Landscape**: Gray/black lines
3. **Energy Levels**: Red dashed lines
4. **Time Counter**: Top-left corner
5. **Statistics**: Summary panel

### Data Access

State vectors are stored in `states_over_time` list:

```python
states, initial_vector = run_simulation(hamiltonian, initial_state, TIME_STEPS, DT)

# Access probability at time step t
prob_density = np.abs(states[t])**2

# Analyze drainage capture
well_indices = range(hole_start, hole_end)
captured_probability = np.sum(prob_density[well_indices])
```

---

## Physical Interpretation

### Why Quantum Mechanics?

1. **Natural Energy Minimization**: Water seeks lowest potential (wells)
2. **Wave-like Spreading**: Water spreads and flows naturally
3. **Momentum Conservation**: Built into Hamiltonian dynamics
4. **Coherent Evolution**: Smooth, continuous dynamics

### Mapping to Hydrology

| Quantum Concept | Hydrological Analog |
|-----------------|---------------------|
| Wave function ψ(x,t) | Water density distribution |
| Potential V(x) | Elevation + infrastructure |
| Kinetic energy -∇² | Flow velocity and momentum |
| Probability \|ψ\|² | Water depth at location |
| Deep well (V<0) | Drainage system capture |
| High barrier (V>0) | Buildings and obstacles |

---

## Troubleshooting

### Common Issues

**1. Animation window not appearing**
- Check matplotlib backend: `plt.get_backend()`
- Try: `export MPLBACKEND=TkAgg`

**2. Simulation too slow**
- Reduce `NUM_QUBITS` (7 instead of 8)
- Reduce `TIME_STEPS`
- Use smaller grid size

**3. No water in drainage**
- Increase `drainage_depth` (more negative)
- Check `vertical_coupling` strength
- Verify hole location overlaps with water path

**4. Water not flowing realistically**
- Increase `gravity_strength`
- Decrease `friction_coeff`
- Check momentum physics enabled

---

## Advanced Usage

### Custom Potential Landscapes

Create your own potential functions:

```python
def custom_potential(x, y):
    # Valley
    V = x**2 + y**2

    # Add drainage at center
    if np.sqrt(x**2 + y**2) < radius:
        V += drainage_depth

    return V
```

### Parameter Sweeps

Automated testing of parameters:

```python
drainage_depths = [-1.0, -3.0, -5.0, -10.0]
for depth in drainage_depths:
    # Run simulation
    # Record capture efficiency
    # Plot results
```

---

## Citation

If you use this V-Space Model in your research, please cite:

```
V-Space Model for Urban Hydrological Systems
Quantum Mechanics-Based Drainage Simulation Framework
[Year] [Author/Institution]
```

---

## License

[Specify license here]

---

## Contact

For questions, issues, or contributions:
- GitHub: [repository link]
- Email: [contact email]

---

## Acknowledgments

This model builds on:
- Quantum mechanics formalism from solid-state physics
- Tight-binding Hamiltonians for lattice systems
- Classical hydrodynamics and momentum conservation
- Urban drainage engineering principles
