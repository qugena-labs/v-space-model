"""
Quick test script for quantum watershed simulation
Run this to verify installation and basic functionality
"""

import sys
import time

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")

    try:
        import numpy as np
        print("  ✓ NumPy")
    except ImportError:
        print("  ✗ NumPy - FAILED")
        return False

    try:
        import scipy
        print("  ✓ SciPy")
    except ImportError:
        print("  ✗ SciPy - FAILED")
        return False

    try:
        import matplotlib
        print("  ✓ Matplotlib")
    except ImportError:
        print("  ✗ Matplotlib - FAILED")
        return False

    try:
        import qiskit
        print(f"  ✓ Qiskit (version {qiskit.__version__})")
    except ImportError:
        print("  ⚠ Qiskit - Not installed (will use classical fallback)")

    return True


def test_simulation_init():
    """Test simulation initialization"""
    print("\nTesting simulation initialization...")

    try:
        from quantum_watershed_simulation import QuantumWatershedSimulation
        sim = QuantumWatershedSimulation(rainfall_type='light')
        print("  ✓ Simulation created successfully")
        print(f"    - Grid: {sim.params.GRID_SIZE}x{sim.params.GRID_SIZE}")
        print(f"    - Buildings: {len(sim.topology.buildings)}")
        print(f"    - Drainage inlets: {len(sim.quantum_system.drainage_inlets)}")
        print(f"    - Quantum mode: {sim.quantum_system.use_quantum}")
        return True
    except Exception as e:
        print(f"  ✗ Simulation initialization - FAILED")
        print(f"    Error: {e}")
        return False


def test_short_run():
    """Test a short simulation run"""
    print("\nTesting short simulation run (10 timesteps)...")

    try:
        from quantum_watershed_simulation import QuantumWatershedSimulation
        sim = QuantumWatershedSimulation(rainfall_type='light')

        start_time = time.time()

        # Run 10 timesteps
        for step in range(10):
            t = step * sim.params.DT
            sim.wave_system.add_rainfall('light')
            sim.wave_system.propagate_step(sim.params.DT)
            sim.quantum_system.update_quantum_state(t)

        elapsed = time.time() - start_time

        stats = sim.wave_system.get_statistics()
        print(f"  ✓ Simulation ran successfully")
        print(f"    - Time: {elapsed:.2f} seconds for 10 steps")
        print(f"    - Total water: {stats['total_water']:.6f}")
        print(f"    - Surface water: {stats['surface_water']:.6f}")

        return True
    except Exception as e:
        print(f"  ✗ Simulation run - FAILED")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis_tools():
    """Test analysis tools can be imported"""
    print("\nTesting analysis tools...")

    try:
        from watershed_analysis_examples import (
            flood_risk_analysis,
            analyze_drainage_effectiveness
        )
        print("  ✓ Analysis tools imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Analysis tools - FAILED")
        print(f"    Error: {e}")
        return False


def test_visualization_setup():
    """Test visualization and animation setup"""
    print("\nTesting visualization setup...")

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        from quantum_watershed_simulation import QuantumWatershedSimulation
        sim = QuantumWatershedSimulation(rainfall_type='light')

        # Setup plots
        sim.visualizer.setup_plots()

        # Verify attributes
        if not hasattr(sim.visualizer, 'fig'):
            raise AttributeError("visualizer.fig not found")
        if not hasattr(sim.visualizer, 'axes'):
            raise AttributeError("visualizer.axes not found")

        # Test update_frame
        frame_data = (0, 0.0)
        sim.visualizer.update_frame(frame_data)

        print("  ✓ Visualization setup successful")
        print("    - Figure created")
        print("    - Axes configured")
        print("    - Frame update works")
        return True
    except Exception as e:
        print(f"  ✗ Visualization setup - FAILED")
        print(f"    Error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("QUANTUM WATERSHED SIMULATION - SYSTEM TEST")
    print("="*60 + "\n")

    all_passed = True

    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install required packages:")
        print("   pip install -r requirements_watershed.txt")
        all_passed = False
        return

    # Test simulation initialization
    if not test_simulation_init():
        all_passed = False

    # Test short run
    if not test_short_run():
        all_passed = False

    # Test analysis tools
    if not test_analysis_tools():
        all_passed = False

    # Test visualization setup
    if not test_visualization_setup():
        all_passed = False

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour simulation is ready to use. Try:")
        print("  python quantum_watershed_simulation.py --rainfall moderate")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the errors above and ensure all dependencies")
        print("are installed correctly.")
    print()


if __name__ == "__main__":
    main()
