# Quick Start Guide: Quantum Watershed Simulation

## 5-Minute Setup

### 1. Install Dependencies

```bash
cd /Users/akshparekh/Documents/qugena-labs/quantumadvantage_modeling
pip install -r requirements_watershed.txt
```

### 2. Run Your First Simulation

```bash
# Basic run with moderate rainfall
python quantum_watershed_simulation.py

# Heavy rainstorm
python quantum_watershed_simulation.py --rainfall heavy

# Fast run without animation
python quantum_watershed_simulation.py --no-animation
```

### 3. What You'll See

The simulation will display 4 real-time panels:

1. **Top-Left**: Surface water distribution with drainage inlets
2. **Top-Right**: Watershed elevation map with buildings and flow vectors
3. **Bottom-Left**: Subsurface drainage layer
4. **Bottom-Right**: Time series of water quantities

### 4. Understanding the Output

At the end, you'll see statistics like:

```
FINAL STATISTICS
Total rainfall added: 0.234567
Total water remaining: 0.123456
  - Surface water: 0.089012
  - Drainage layer: 0.034444
Lakebed accumulation: 0.045678
Total infiltrated: 0.056789
Total drained: 0.012345
```

**Key Metrics:**
- **Surface water**: Visible water on roads, parks, etc.
- **Drainage layer**: Water in underground storm sewers
- **Lakebed accumulation**: Water collected in low-lying basin
- **Infiltrated**: Water absorbed into soil (quantum-controlled)
- **Drained**: Water removed via drainage system

### 5. Run Analysis Examples

```bash
# Compare all rainfall scenarios
python watershed_analysis_examples.py --analysis compare

# Flood risk assessment
python watershed_analysis_examples.py --analysis flood

# Green infrastructure impact
python watershed_analysis_examples.py --analysis green

# Storm surge scenario
python watershed_analysis_examples.py --analysis surge

# Run all analyses
python watershed_analysis_examples.py --analysis all
```

### 6. Interactive Exploration (Jupyter)

```bash
jupyter notebook watershed_interactive_notebook.ipynb
```

Then run cells sequentially to:
- Visualize the watershed topology
- Run custom simulations
- Analyze quantum states
- Export results

## Common Use Cases

### Testing Different Rain Intensities

```bash
python quantum_watershed_simulation.py --rainfall light     # 5 mm/hour
python quantum_watershed_simulation.py --rainfall moderate  # 15 mm/hour
python quantum_watershed_simulation.py --rainfall heavy     # 30 mm/hour
python quantum_watershed_simulation.py --rainfall extreme   # 80 mm/hour
```

### Quick Testing (Faster)

Edit `quantum_watershed_simulation.py` line 76-80:

```python
GRID_SIZE = 100        # Reduce from 150
TOTAL_TIME = 1800.0    # Reduce from 3600 (30 min instead of 1 hour)
```

### Saving Animation

```bash
# Requires ffmpeg installed
python quantum_watershed_simulation.py --save-animation
# Creates: watershed_simulation.mp4
```

## Key Features Explained

### Quantum Control

The simulation uses **56 total qubits** across 5 registers:
- **Zone 1 (Hills)**: 7 qubits controlling infiltration
- **Zone 2 (Suburban)**: 10 qubits
- **Zone 3 (Urban)**: 15 qubits
- **Park**: 4 qubits for green infrastructure
- **Drainage**: 8 qubits controlling storm sewer inlets

Each qubit measurement determines:
- **|0⟩**: Infiltration disabled / Inlet blocked
- **|1⟩**: Infiltration active / Inlet open

### Physical Realism

- **Gravity**: Water flows downhill following elevation gradient
- **Uphill lag**: Slower propagation uphill vs downhill
- **Mass conservation**: Total probability strictly normalized to 1
- **Zone-specific behavior**: Different dispersion rates in each zone
- **Building barriers**: Tall buildings create potential obstacles

### Infrastructure

- **3 Zones**: Hills (residential) → Suburban (park) → Urban/Lakebed
- **~30 Buildings**: Houses, skyscrapers, park tree
- **5 Road segments**: Curved roads with sidewalks
- **~40 Drainage inlets**: Quantum-controlled storm sewer access

## Troubleshooting

### "ModuleNotFoundError: No module named 'qiskit'"

```bash
pip install qiskit qiskit-aer
```

### Simulation runs slowly

Option 1: Run without animation:
```bash
python quantum_watershed_simulation.py --no-animation
```

Option 2: Reduce grid size (edit line 76 in `quantum_watershed_simulation.py`):
```python
GRID_SIZE = 100  # Instead of 150
```

### No plot window appears

For headless systems, switch matplotlib backend:

Edit top of `quantum_watershed_simulation.py`:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

### "Qiskit not available" warning

The simulation will automatically fall back to classical random sampling. All features work, but without true quantum behavior. To fix, install Qiskit.

## Next Steps

1. **Read the full README**: `README_WATERSHED.md`
2. **Explore analysis tools**: `watershed_analysis_examples.py`
3. **Try the Jupyter notebook**: `watershed_interactive_notebook.ipynb`
4. **Modify parameters**: Edit `SimulationParameters` class
5. **Add custom scenarios**: See Section 11 in README

## File Overview

```
quantumadvantage_modeling/
├── quantum_watershed_simulation.py          # Main simulation (run this)
├── watershed_analysis_examples.py           # Analysis tools
├── watershed_interactive_notebook.ipynb     # Interactive exploration
├── requirements_watershed.txt               # Python dependencies
├── README_WATERSHED.md                      # Full documentation
└── QUICKSTART_WATERSHED.md                  # This file
```

## Example Workflow

```bash
# 1. Install
pip install -r requirements_watershed.txt

# 2. Quick test run (moderate rain)
python quantum_watershed_simulation.py --rainfall moderate

# 3. Analyze flood risk
python watershed_analysis_examples.py --analysis flood

# 4. Compare scenarios
python watershed_analysis_examples.py --analysis compare

# 5. Explore interactively
jupyter notebook watershed_interactive_notebook.ipynb
```

## Getting Help

- Check `README_WATERSHED.md` for detailed documentation
- Review code comments in `quantum_watershed_simulation.py`
- Try the Jupyter notebook for interactive examples
- Examine `watershed_analysis_examples.py` for analysis patterns

---

**Ready to simulate? Run this now:**

```bash
python quantum_watershed_simulation.py --rainfall moderate
```

Expected runtime: ~5-10 minutes
