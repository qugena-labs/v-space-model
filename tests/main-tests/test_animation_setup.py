"""
Test animation setup to verify the fig attribute fix
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from quantum_watershed_simulation import QuantumWatershedSimulation

print("Testing animation setup...")
print()

# Create simulation
sim = QuantumWatershedSimulation(rainfall_type='light')

# Setup plots (this is what happens before animation starts)
sim.visualizer.setup_plots()

# Verify fig attribute exists in visualizer
if hasattr(sim.visualizer, 'fig'):
    print("✓ sim.visualizer.fig exists")
else:
    print("✗ sim.visualizer.fig is missing")
    exit(1)

# Verify axes exist
if hasattr(sim.visualizer, 'axes'):
    print("✓ sim.visualizer.axes exists")
else:
    print("✗ sim.visualizer.axes is missing")
    exit(1)

# Test that update_frame can be called
try:
    frame_data = (0, 0.0)
    result = sim.visualizer.update_frame(frame_data)
    print("✓ update_frame() works correctly")
except Exception as e:
    print(f"✗ update_frame() failed: {e}")
    exit(1)

print()
print("✅ Animation setup test passed!")
print()
print("The simulation is ready to run with animation.")
