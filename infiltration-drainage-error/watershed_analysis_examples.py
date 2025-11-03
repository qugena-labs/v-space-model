"""
Example Analysis Scripts for Quantum Watershed Simulation
==========================================================

Additional analysis tools and example scenarios for the watershed simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_watershed_simulation import (
    QuantumWatershedSimulation,
    SimulationParameters,
    WatershedTopology,
    QuantumInfiltrationSystem,
    WaveEvolution
)


# ============================================================================
# FLOOD RISK ANALYSIS
# ============================================================================

def flood_risk_analysis(simulation: QuantumWatershedSimulation, threshold: float = 0.005):
    """
    Analyze flood risk across the watershed.

    Parameters:
    -----------
    simulation : QuantumWatershedSimulation
        Completed simulation instance
    threshold : float
        Water density threshold for flood classification

    Returns:
    --------
    flood_map : np.ndarray
        Boolean array indicating flood risk areas
    """
    density = simulation.wave_system.get_water_density()
    flood_map = density > threshold

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Water density
    im1 = axes[0].imshow(density, cmap='Blues', origin='lower')
    axes[0].set_title('Water Density Distribution')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    plt.colorbar(im1, ax=axes[0], label='Density')

    # Panel 2: Flood risk zones
    im2 = axes[1].imshow(flood_map, cmap='RdYlGn_r', origin='lower')
    axes[1].set_title(f'Flood Risk Map (threshold={threshold})')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    plt.colorbar(im2, ax=axes[1], label='At Risk')

    # Overlay buildings on both
    for bldg in simulation.topology.buildings:
        for ax in axes:
            from matplotlib.patches import Rectangle
            rect = Rectangle((bldg['x'], bldg['y']), bldg['size'], bldg['size'],
                           linewidth=1, edgecolor='red', facecolor='none', alpha=0.7)
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig('flood_risk_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved flood risk analysis to flood_risk_analysis.png")

    # Calculate statistics
    total_cells = flood_map.size
    flooded_cells = np.sum(flood_map)
    flood_percentage = (flooded_cells / total_cells) * 100

    print("\nFlood Risk Statistics:")
    print(f"  Total area: {total_cells} cells")
    print(f"  Flooded area: {flooded_cells} cells ({flood_percentage:.2f}%)")

    # Flood risk by zone
    zone_mask = simulation.topology.zone_mask
    for zone_id in [1, 2, 3]:
        zone_name = ['Hills', 'Suburban', 'Urban/Lakebed'][zone_id - 1]
        zone_cells = np.sum(zone_mask == zone_id)
        zone_flooded = np.sum((zone_mask == zone_id) & flood_map)
        zone_pct = (zone_flooded / zone_cells) * 100 if zone_cells > 0 else 0
        print(f"  Zone {zone_id} ({zone_name}): {zone_flooded}/{zone_cells} cells ({zone_pct:.2f}%)")

    return flood_map


# ============================================================================
# COMPARATIVE SCENARIO ANALYSIS
# ============================================================================

def compare_rainfall_scenarios():
    """
    Compare different rainfall scenarios and their impacts.
    """
    scenarios = ['light', 'moderate', 'heavy', 'extreme']
    results = {}

    print("\n" + "="*70)
    print("COMPARATIVE SCENARIO ANALYSIS")
    print("="*70 + "\n")

    for scenario in scenarios:
        print(f"Running scenario: {scenario}")

        # Create simulation
        sim = QuantumWatershedSimulation(rainfall_type=scenario)

        # Run without animation for speed
        for step in range(sim.params.N_TIMESTEPS):
            t = step * sim.params.DT
            sim.wave_system.add_rainfall(scenario)
            sim.wave_system.propagate_step(sim.params.DT)
            sim.quantum_system.update_quantum_state(t)

            if step % 200 == 0:
                print(f"  Progress: {(step/sim.params.N_TIMESTEPS)*100:.0f}%", end='\r')

        print(f"  Completed {scenario:10s}")

        # Collect statistics
        stats = sim.wave_system.get_statistics()
        results[scenario] = stats

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rainfall Scenario Comparison', fontsize=16, fontweight='bold')

    metrics = ['surface_water', 'drainage_water', 'lakebed_accumulation', 'total_infiltrated']
    titles = ['Surface Water', 'Drainage Layer', 'Lakebed Collection', 'Infiltration']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        values = [results[scenario][metric] for scenario in scenarios]
        colors = ['lightblue', 'blue', 'darkblue', 'navy']

        bars = ax.bar(scenarios, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Normalized Quantity')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('scenario_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison to scenario_comparison.png")

    return results


# ============================================================================
# GREEN INFRASTRUCTURE IMPACT
# ============================================================================

def evaluate_green_infrastructure():
    """
    Compare simulation with and without enhanced park infiltration.
    """
    print("\n" + "="*70)
    print("GREEN INFRASTRUCTURE IMPACT ANALYSIS")
    print("="*70 + "\n")

    # Baseline scenario
    print("Running baseline scenario (current park capacity)...")
    sim_baseline = QuantumWatershedSimulation(rainfall_type='heavy')

    for step in range(sim_baseline.params.N_TIMESTEPS):
        t = step * sim_baseline.params.DT
        sim_baseline.wave_system.add_rainfall('heavy')
        sim_baseline.wave_system.propagate_step(sim_baseline.params.DT)
        sim_baseline.quantum_system.update_quantum_state(t)

        if step % 200 == 0:
            print(f"  Progress: {(step/sim_baseline.params.N_TIMESTEPS)*100:.0f}%", end='\r')

    print("  Completed baseline scenario")
    stats_baseline = sim_baseline.wave_system.get_statistics()

    # Enhanced green infrastructure scenario
    print("\nRunning enhanced scenario (increased park infiltration)...")

    # Modify parameters for enhanced scenario
    original_park_infiltration = SimulationParameters.INFILTRATION_DEPTH_PARK
    SimulationParameters.INFILTRATION_DEPTH_PARK = 0.80  # Increase from 0.40 to 0.80

    sim_enhanced = QuantumWatershedSimulation(rainfall_type='heavy')

    for step in range(sim_enhanced.params.N_TIMESTEPS):
        t = step * sim_enhanced.params.DT
        sim_enhanced.wave_system.add_rainfall('heavy')
        sim_enhanced.wave_system.propagate_step(sim_enhanced.params.DT)
        sim_enhanced.quantum_system.update_quantum_state(t)

        if step % 200 == 0:
            print(f"  Progress: {(step/sim_enhanced.params.N_TIMESTEPS)*100:.0f}%", end='\r')

    print("  Completed enhanced scenario")
    stats_enhanced = sim_enhanced.wave_system.get_statistics()

    # Restore original parameter
    SimulationParameters.INFILTRATION_DEPTH_PARK = original_park_infiltration

    # Comparison
    print("\n" + "-"*70)
    print("RESULTS COMPARISON")
    print("-"*70)

    metrics_to_compare = [
        ('surface_water', 'Surface Water Remaining'),
        ('lakebed_accumulation', 'Lakebed Accumulation'),
        ('total_infiltrated', 'Total Infiltration')
    ]

    for metric, label in metrics_to_compare:
        baseline_val = stats_baseline[metric]
        enhanced_val = stats_enhanced[metric]
        change = ((enhanced_val - baseline_val) / baseline_val) * 100 if baseline_val > 0 else 0

        print(f"\n{label}:")
        print(f"  Baseline:  {baseline_val:.6f}")
        print(f"  Enhanced:  {enhanced_val:.6f}")
        print(f"  Change:    {change:+.2f}%")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['surface_water', 'drainage_water', 'lakebed_accumulation', 'total_infiltrated']
    labels = ['Surface\nWater', 'Drainage\nLayer', 'Lakebed\nCollection', 'Total\nInfiltration']

    baseline_values = [stats_baseline[m] for m in metrics]
    enhanced_values = [stats_enhanced[m] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (40cm park infiltration)',
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, enhanced_values, width, label='Enhanced (80cm park infiltration)',
                   color='forestgreen', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Normalized Quantity', fontweight='bold')
    ax.set_title('Green Infrastructure Impact on Watershed Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('green_infrastructure_impact.png', dpi=150, bbox_inches='tight')
    print("\nSaved analysis to green_infrastructure_impact.png")


# ============================================================================
# DRAINAGE SYSTEM EFFECTIVENESS
# ============================================================================

def analyze_drainage_effectiveness(simulation: QuantumWatershedSimulation):
    """
    Analyze the effectiveness of the drainage system.
    """
    print("\n" + "="*70)
    print("DRAINAGE SYSTEM EFFECTIVENESS ANALYSIS")
    print("="*70 + "\n")

    stats = simulation.wave_system.get_statistics()
    drainage_status = simulation.quantum_system.get_drainage_status()

    # Count functional vs blocked inlets
    total_inlets = len(drainage_status)
    functional_inlets = sum(drainage_status.values())
    blocked_inlets = total_inlets - functional_inlets

    print("Drainage Infrastructure Status:")
    print(f"  Total inlets: {total_inlets}")
    print(f"  Functional: {functional_inlets} ({(functional_inlets/total_inlets)*100:.1f}%)")
    print(f"  Blocked: {blocked_inlets} ({(blocked_inlets/total_inlets)*100:.1f}%)")
    print()

    print("Water Management:")
    print(f"  Surface water: {stats['surface_water']:.6f}")
    print(f"  Drainage layer: {stats['drainage_water']:.6f}")
    print(f"  Total drained: {stats['total_drained']:.6f}")

    if stats['total_rainfall'] > 0:
        drainage_efficiency = (stats['total_drained'] / stats['total_rainfall']) * 100
        print(f"  Drainage efficiency: {drainage_efficiency:.2f}%")

    # Spatial distribution of drainage
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Inlet locations colored by status
    ax1 = axes[0]

    # Show base topology
    ax1.imshow(simulation.topology.elevation, cmap='terrain', origin='lower', alpha=0.3)

    # Mark inlets
    for (i, j), is_open in drainage_status.items():
        color = 'lime' if is_open else 'red'
        marker = 'o' if is_open else 'x'
        ax1.plot(i, j, marker, color=color, markersize=8, markeredgewidth=2)

    ax1.set_title('Drainage Inlet Status', fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend(['Functional', 'Blocked'], loc='upper right')

    # Panel 2: Drainage layer water distribution
    ax2 = axes[1]
    drainage_density = simulation.wave_system.get_drainage_density()
    im2 = ax2.imshow(drainage_density, cmap='Purples', origin='lower')
    ax2.set_title('Subsurface Water Distribution', fontweight='bold')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax2, label='Water Density')

    plt.tight_layout()
    plt.savefig('drainage_effectiveness.png', dpi=150, bbox_inches='tight')
    print("\nSaved analysis to drainage_effectiveness.png")


# ============================================================================
# STORM SURGE SCENARIO
# ============================================================================

def storm_surge_scenario():
    """
    Simulate a storm surge event (sudden intense rainfall).
    """
    print("\n" + "="*70)
    print("STORM SURGE SCENARIO")
    print("="*70 + "\n")
    print("Simulating: Moderate rain with 3-minute extreme burst\n")

    sim = QuantumWatershedSimulation(rainfall_type='moderate')

    # Track water levels over time
    time_points = []
    surface_water = []

    for step in range(sim.params.N_TIMESTEPS):
        t = step * sim.params.DT

        # Storm surge: extreme rain between 600-780 seconds (10-13 minutes)
        if 600 <= t <= 780:
            sim.wave_system.add_rainfall('extreme')
        else:
            sim.wave_system.add_rainfall('moderate')

        sim.wave_system.propagate_step(sim.params.DT)
        sim.quantum_system.update_quantum_state(t)

        # Record every 30 seconds
        if step % 15 == 0:
            stats = sim.wave_system.get_statistics()
            time_points.append(t)
            surface_water.append(stats['surface_water'])

        if step % 200 == 0:
            print(f"  Progress: {(step/sim.params.N_TIMESTEPS)*100:.0f}%", end='\r')

    print("  Completed simulation")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(np.array(time_points) / 60, surface_water, 'b-', linewidth=2, label='Surface Water')

    # Shade storm surge period
    ax.axvspan(600/60, 780/60, alpha=0.2, color='red', label='Storm Surge Period')

    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Surface Water Quantity', fontweight='bold')
    ax.set_title('Storm Surge Impact on Surface Water Levels', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('storm_surge_scenario.png', dpi=150, bbox_inches='tight')
    print("\nSaved analysis to storm_surge_scenario.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all example analyses"""
    import argparse

    parser = argparse.ArgumentParser(description='Watershed Simulation Analysis Examples')
    parser.add_argument('--analysis', type=str, default='all',
                       choices=['all', 'flood', 'compare', 'green', 'drainage', 'surge'],
                       help='Which analysis to run')

    args = parser.parse_args()

    if args.analysis in ['all', 'compare']:
        print("\n>>> Running Comparative Scenario Analysis...")
        compare_rainfall_scenarios()

    if args.analysis in ['all', 'green']:
        print("\n>>> Running Green Infrastructure Analysis...")
        evaluate_green_infrastructure()

    if args.analysis in ['all', 'surge']:
        print("\n>>> Running Storm Surge Scenario...")
        storm_surge_scenario()

    if args.analysis in ['flood', 'drainage']:
        print("\n>>> Running simulation for detailed analysis...")
        sim = QuantumWatershedSimulation(rainfall_type='heavy')

        # Run simulation
        for step in range(sim.params.N_TIMESTEPS):
            t = step * sim.params.DT
            sim.wave_system.add_rainfall('heavy')
            sim.wave_system.propagate_step(sim.params.DT)
            sim.quantum_system.update_quantum_state(t)

            if step % 200 == 0:
                print(f"  Progress: {(step/sim.params.N_TIMESTEPS)*100:.0f}%", end='\r')

        print("  Completed simulation")

        if args.analysis in ['all', 'flood']:
            print("\n>>> Running Flood Risk Analysis...")
            flood_risk_analysis(sim)

        if args.analysis in ['all', 'drainage']:
            print("\n>>> Running Drainage Effectiveness Analysis...")
            analyze_drainage_effectiveness(sim)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - flood_risk_analysis.png (if applicable)")
    print("  - scenario_comparison.png (if applicable)")
    print("  - green_infrastructure_impact.png (if applicable)")
    print("  - drainage_effectiveness.png (if applicable)")
    print("  - storm_surge_scenario.png (if applicable)")
    print()


if __name__ == "__main__":
    main()
