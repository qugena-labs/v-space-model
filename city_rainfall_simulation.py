"""
City Rainfall Wave Simulation
==============================

Simulates rainfall as waves flowing down the city landscape with:
- Rain source in western region (middle third)
- Wave diffusion and gravity-driven flow
- Infiltration in micro-wells
- Drainage system capture
- Real-time animation of cross-section (middle view) and bird's eye
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import landscape class - use the static version
import importlib.util
spec = importlib.util.spec_from_file_location("static_city_potential",
                                                os.path.join(os.path.dirname(__file__), "static_city_potential.py"))
simple_city = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simple_city)
CityPotentialLandscape = simple_city.CityPotentialLandscape

class RainfallWaveSimulation:
    """Simulates rainfall as waves on the city landscape"""

    def __init__(self, landscape: CityPotentialLandscape, rainfall_intensity: str = 'moderate', duration: str = 'standard'):
        self.landscape = landscape
        self.nx = landscape.nx
        self.ny = landscape.ny

        # Rainfall intensities (amplitude units per timestep) - 6 levels
        self.rainfall_rates = {
            'drizzle': 0.03,
            'light': 0.08,
            'moderate': 0.18,
            'heavy': 0.35,
            'severe': 0.60,
            'extreme': 1.00
        }
        self.rainfall_rate = self.rainfall_rates.get(rainfall_intensity, 0.18)
        self.rainfall_intensity = rainfall_intensity

        # Water wave field (height of water at each point)
        self.water = np.zeros((self.ny, self.nx))

        # Velocity fields for momentum (persistent across timesteps)
        self.velocity_x = np.zeros((self.ny, self.nx))
        self.velocity_y = np.zeros((self.ny, self.nx))

        # Drainage system water and velocity fields
        self.drainage_water = np.zeros((self.ny, self.nx))
        self.drainage_velocity_x = np.zeros((self.ny, self.nx))
        self.drainage_velocity_y = np.zeros((self.ny, self.nx))

        # Duration scenarios (total_time, rain_duration, description)
        duration_configs = {
            'standard': (100.0, 50.0, 'Standard storm (rain for half)'),
            'storm': (150.0, 150.0, 'Continuous storm'),
            'day': (300.0, 300.0, 'Continuous 1-day rain'),
            'week': (700.0, 700.0, 'Continuous 7-day rain'),
            '10day': (1000.0, 1000.0, 'Continuous 10-day rain'),
            'month': (1500.0, 1500.0, 'Continuous 30-day rain')
        }

        config = duration_configs.get(duration, (100.0, 50.0, 'Standard storm'))
        self.duration_type = duration
        self.duration_description = config[2]

        # Simulation parameters
        self.dt = 0.1  # Time step
        self.t = 0.0
        self.max_time = config[0]  # Total simulation time
        self.rain_duration = config[1]  # How long rain lasts

        # Wave physics parameters
        self.diffusion_coeff = 0.05  # Water spreads out
        self.gravity_strength = 0.8  # Gravity acceleration down slope (F = -grad(potential))
        self.friction_coeff = 0.15  # Friction coefficient (slows water down)
        self.infiltration_rate = 0.02  # Water absorbed per timestep
        self.drainage_capture_rate = 0.3  # How much drainage captures

        # Rainfall region (west area, middle third vertically)
        self.rain_x_start = 0.0
        self.rain_x_end = 0.15  # First 15% of domain
        self.rain_y_start = 0.33  # Middle third starts
        self.rain_y_end = 0.67  # Middle third ends

        # Statistics
        self.total_rain_added = 0.0
        self.total_infiltrated = 0.0
        self.total_drained = 0.0

        print(f"\nRainfall Wave Simulation Initialized")
        print(f"  Intensity: {rainfall_intensity} ({self.rainfall_rate:.2f} units/step)")
        print(f"  Duration: {self.duration_description}")
        print(f"  Simulation time: {self.max_time:.1f} time units")
        print(f"  Rain duration: {self.rain_duration:.1f} time units")
        print(f"  Rain zone: x={self.rain_x_start:.0%} to {self.rain_x_end:.0%}, y={self.rain_y_start:.0%} to {self.rain_y_end:.0%}")

    def add_rainfall(self):
        """Add rain to the western region (middle third)"""
        if self.t > self.rain_duration:
            return  # No more rain after half time

        # Define rain region
        x_start_idx = int(self.rain_x_start * self.nx)
        x_end_idx = int(self.rain_x_end * self.nx)
        y_start_idx = int(self.rain_y_start * self.ny)
        y_end_idx = int(self.rain_y_end * self.ny)

        # Add rain with Gaussian distribution (concentrated in center)
        for i in range(y_start_idx, y_end_idx):
            for j in range(x_start_idx, x_end_idx):
                # Gaussian falloff from center
                center_x = (x_start_idx + x_end_idx) / 2
                center_y = (y_start_idx + y_end_idx) / 2
                dist = np.sqrt((j - center_x)**2 + (i - center_y)**2)
                intensity = np.exp(-dist**2 / 500.0)

                # Add rain (amplitude increases over time for first half)
                rain_amount = self.rainfall_rate * intensity
                self.water[i, j] += rain_amount
                self.total_rain_added += rain_amount

    def compute_gradient(self):
        """Compute gradient of potential (for gravity flow direction)"""
        # Gradient of surface potential
        grad_y, grad_x = np.gradient(self.landscape.surface_potential)
        return grad_x, grad_y

    def step(self):
        """Advance simulation by one timestep with momentum conservation"""
        # Add rainfall
        self.add_rainfall()

        # Get potential gradient (downhill direction)
        grad_x, grad_y = self.compute_gradient()

        # ===== MOMENTUM UPDATE =====
        # Apply gravity force: F = -grad(potential)
        # dv/dt = F - friction*v
        # Only update velocity where water exists
        where_water = self.water > 0.01

        if np.any(where_water):
            # Gravity force (proportional to slope)
            force_x = -self.gravity_strength * grad_x
            force_y = -self.gravity_strength * grad_y

            # Friction force (opposes motion)
            friction_x = -self.friction_coeff * self.velocity_x
            friction_y = -self.friction_coeff * self.velocity_y

            # Update velocity: v_new = v_old + (F - friction*v) * dt
            self.velocity_x[where_water] += (force_x[where_water] + friction_x[where_water]) * self.dt
            self.velocity_y[where_water] += (force_y[where_water] + friction_y[where_water]) * self.dt

            # Cap maximum velocity to prevent instabilities
            max_vel = 5.0
            vel_magnitude = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
            vel_too_fast = vel_magnitude > max_vel
            if np.any(vel_too_fast):
                scale = max_vel / vel_magnitude[vel_too_fast]
                self.velocity_x[vel_too_fast] *= scale
                self.velocity_y[vel_too_fast] *= scale

        # Where no water exists, gradually decay velocity
        no_water = self.water <= 0.01
        self.velocity_x[no_water] *= 0.9
        self.velocity_y[no_water] *= 0.9

        # ===== ADVECTION STEP (move water according to velocity) =====
        water_new = self.water.copy()

        # Simple upwind advection using velocity field
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if self.water[i, j] > 0.01:
                    vx = self.velocity_x[i, j]
                    vy = self.velocity_y[i, j]

                    # Determine upwind direction and transport water
                    if vx > 0:
                        flux_x = vx * self.water[i, j]
                        if j + 1 < self.nx:
                            water_new[i, j] -= flux_x * self.dt
                            water_new[i, j + 1] += flux_x * self.dt
                    elif vx < 0:
                        flux_x = -vx * self.water[i, j]
                        if j - 1 >= 0:
                            water_new[i, j] -= flux_x * self.dt
                            water_new[i, j - 1] += flux_x * self.dt

                    if vy > 0:
                        flux_y = vy * self.water[i, j]
                        if i + 1 < self.ny:
                            water_new[i, j] -= flux_y * self.dt
                            water_new[i + 1, j] += flux_y * self.dt
                    elif vy < 0:
                        flux_y = -vy * self.water[i, j]
                        if i - 1 >= 0:
                            water_new[i, j] -= flux_y * self.dt
                            water_new[i - 1, j] += flux_y * self.dt

        self.water = water_new

        # Diffusion step (water spreads out)
        laplacian = (
            np.roll(self.water, 1, axis=0) +
            np.roll(self.water, -1, axis=0) +
            np.roll(self.water, 1, axis=1) +
            np.roll(self.water, -1, axis=1) -
            4 * self.water
        )
        self.water += self.diffusion_coeff * laplacian * self.dt

        # Infiltration (water absorbed into wells)
        infiltration = np.minimum(self.water, self.infiltration_rate * self.dt)
        self.water -= infiltration
        self.total_infiltrated += np.sum(infiltration)

        # Drainage capture (at surface openings) - transfer to drainage system
        for ox, oy in self.landscape.surface_openings:
            oi = int(oy * self.ny)
            oj = int(ox * self.nx)
            if 0 <= oi < self.ny and 0 <= oj < self.nx:
                # Capture water from surface into drainage
                captured = self.water[oi, oj] * self.drainage_capture_rate
                self.water[oi, oj] -= captured
                self.drainage_water[oi, oj] += captured  # Add to drainage system
                self.total_drained += captured

        # ===== DRAINAGE SYSTEM MOMENTUM FLOW =====
        # Water in drainage flows with momentum (similar to surface)

        # Get drainage potential gradient
        grad_drain_y, grad_drain_x = np.gradient(self.landscape.drainage_potential)

        # Where drainage water exists
        where_drain_water = self.drainage_water > 0.01

        if np.any(where_drain_water):
            # Apply gravity force in drainage channels
            force_drain_x = -self.gravity_strength * 1.5 * grad_drain_x  # 1.5x faster in pipes
            force_drain_y = -self.gravity_strength * 1.5 * grad_drain_y

            # Lower friction in drainage (smoother pipes)
            friction_drain_x = -self.friction_coeff * 0.5 * self.drainage_velocity_x
            friction_drain_y = -self.friction_coeff * 0.5 * self.drainage_velocity_y

            # Update drainage velocity
            self.drainage_velocity_x[where_drain_water] += (force_drain_x[where_drain_water] +
                                                             friction_drain_x[where_drain_water]) * self.dt
            self.drainage_velocity_y[where_drain_water] += (force_drain_y[where_drain_water] +
                                                             friction_drain_y[where_drain_water]) * self.dt

            # Cap drainage velocity
            max_drain_vel = 8.0  # Faster than surface
            drain_vel_mag = np.sqrt(self.drainage_velocity_x**2 + self.drainage_velocity_y**2)
            drain_too_fast = drain_vel_mag > max_drain_vel
            if np.any(drain_too_fast):
                scale = max_drain_vel / drain_vel_mag[drain_too_fast]
                self.drainage_velocity_x[drain_too_fast] *= scale
                self.drainage_velocity_y[drain_too_fast] *= scale

        # Decay velocity where no drainage water
        no_drain_water = self.drainage_water <= 0.01
        self.drainage_velocity_x[no_drain_water] *= 0.85
        self.drainage_velocity_y[no_drain_water] *= 0.85

        # Advect drainage water using velocity
        drainage_water_new = self.drainage_water.copy()
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if self.drainage_water[i, j] > 0.01:
                    vx = self.drainage_velocity_x[i, j]
                    vy = self.drainage_velocity_y[i, j]

                    if vx > 0:
                        flux_x = vx * self.drainage_water[i, j]
                        if j + 1 < self.nx:
                            drainage_water_new[i, j] -= flux_x * self.dt
                            drainage_water_new[i, j + 1] += flux_x * self.dt
                    elif vx < 0:
                        flux_x = -vx * self.drainage_water[i, j]
                        if j - 1 >= 0:
                            drainage_water_new[i, j] -= flux_x * self.dt
                            drainage_water_new[i, j - 1] += flux_x * self.dt

                    if vy > 0:
                        flux_y = vy * self.drainage_water[i, j]
                        if i + 1 < self.ny:
                            drainage_water_new[i, j] -= flux_y * self.dt
                            drainage_water_new[i + 1, j] += flux_y * self.dt
                    elif vy < 0:
                        flux_y = -vy * self.drainage_water[i, j]
                        if i - 1 >= 0:
                            drainage_water_new[i, j] -= flux_y * self.dt
                            drainage_water_new[i - 1, j] += flux_y * self.dt

        self.drainage_water = drainage_water_new

        # Drainage water exits at lakebed (east end)
        lakebed_x_idx = int(0.70 * self.nx)
        self.drainage_water[:, lakebed_x_idx:] *= 0.8  # Water exits drainage system

        # Prevent negative water
        self.water = np.maximum(self.water, 0.0)
        self.drainage_water = np.maximum(self.drainage_water, 0.0)

        # Surface water reaching lakebed is removed (absorbed)
        self.water[:, lakebed_x_idx:] *= 0.9  # Gradual removal in lakebed area

        # Advance time
        self.t += self.dt

    def run_simulation(self):
        """Run the full simulation with animation"""
        print("\nStarting rainfall wave simulation...")
        print("Close the animation window to finish.\n")

        # Setup figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Initialize plots
        def init():
            return []

        def update(frame):
            # Run 5 steps per frame for smoother animation
            for _ in range(5):
                self.step()

            # Clear axes
            ax1.clear()
            ax2.clear()

            # ===== PLOT 1: BIRD'S EYE VIEW =====
            # Show water from above
            water_display = np.log10(self.water + 0.01)  # Log scale for visibility

            # Show terrain as background
            terrain_img = ax1.imshow(self.landscape.surface_potential,
                                    extent=[0, 1, 0, 1],
                                    origin='lower',
                                    cmap='terrain',
                                    alpha=0.3,
                                    aspect='auto')

            # Overlay water
            water_img = ax1.imshow(water_display,
                                  extent=[0, 1, 0, 1],
                                  origin='lower',
                                  cmap='Blues',
                                  alpha=0.7,
                                  vmin=-2, vmax=1,
                                  aspect='auto')

            # Mark rain zone
            ax1.axvline(self.rain_x_end, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax1.axhline(self.rain_y_start, color='red', linestyle='--', alpha=0.3)
            ax1.axhline(self.rain_y_end, color='red', linestyle='--', alpha=0.3)

            # Mark drainage openings
            if self.landscape.surface_openings:
                ox = [o[0] for o in self.landscape.surface_openings]
                oy = [o[1] for o in self.landscape.surface_openings]
                ax1.scatter(ox, oy, c='yellow', s=10, marker='o', edgecolors='black', linewidths=0.5)

            ax1.set_xlabel('West ← → East', fontsize=12)
            ax1.set_ylabel('South ← → North', fontsize=12)
            rain_status = "RAINING" if self.t < self.rain_duration else "DRY"
            ax1.set_title(f'BIRD\'S EYE VIEW - Water Flow | t={self.t:.1f} | {rain_status}',
                         fontsize=14, fontweight='bold')

            # ===== PLOT 2: WAVE AMPLITUDE VIEW =====
            # Show water amplitude (wave height) with MOMENTUM flow vectors

            # Plot water amplitude as heatmap
            water_amp = np.sqrt(self.water)  # Amplitude (sqrt of intensity)
            water_log = np.log10(self.water + 0.001)  # Log scale for better visibility

            amp_img = ax2.imshow(water_log,
                                extent=[0, 1, 0, 1],
                                origin='lower',
                                cmap='hot',
                                alpha=0.9,
                                vmin=-3, vmax=0.5,
                                aspect='auto')

            # Overlay MOMENTUM flow vectors (actual velocity field with inertia)
            skip = 8  # Subsample for arrows
            X_sub, Y_sub = np.meshgrid(
                self.landscape.x[::skip],
                self.landscape.y[::skip]
            )
            water_sub = self.water[::skip, ::skip]
            # Use actual velocity field (this shows momentum!)
            vx_sub = self.velocity_x[::skip, ::skip]
            vy_sub = self.velocity_y[::skip, ::skip]

            # Only show arrows where water exists
            mask = water_sub > 0.05
            if np.any(mask):
                ax2.quiver(X_sub[mask], Y_sub[mask],
                          vx_sub[mask], vy_sub[mask],
                          color='cyan',
                          scale=15,
                          width=0.003,
                          alpha=0.7)

            # Mark drainage usage (where water is being drained)
            drainage_active = []
            for ox, oy in self.landscape.surface_openings:
                oi = int(oy * self.ny)
                oj = int(ox * self.nx)
                if 0 <= oi < self.ny and 0 <= oj < self.nx:
                    if self.water[oi, oj] > 0.01:  # Drainage is active here
                        drainage_active.append((ox, oy))

            if drainage_active:
                dx = [d[0] for d in drainage_active]
                dy = [d[1] for d in drainage_active]
                ax2.scatter(dx, dy, c='lime', s=50, marker='s',
                           edgecolors='black', linewidths=1.5,
                           label='Active Drainage', zorder=10)

            # Mark rain zone
            from matplotlib.patches import Rectangle
            rain_rect = Rectangle((self.rain_x_start, self.rain_y_start),
                                 self.rain_x_end - self.rain_x_start,
                                 self.rain_y_end - self.rain_y_start,
                                 linewidth=2, edgecolor='red',
                                 facecolor='none', linestyle='--')
            ax2.add_patch(rain_rect)

            ax2.set_xlabel('West ← → East', fontsize=12)
            ax2.set_ylabel('South ← → North', fontsize=12)
            ax2.set_title(f'WAVE AMPLITUDE & MOMENTUM VECTORS | Arrows show velocity with inertia',
                         fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            if drainage_active:
                ax2.legend(loc='upper right')

            # Add statistics text
            stats_text = (f'Rain Added: {self.total_rain_added:.1f}\n'
                         f'Infiltrated: {self.total_infiltrated:.1f}\n'
                         f'Drained: {self.total_drained:.1f}\n'
                         f'On Surface: {np.sum(self.water):.1f}')
            ax2.text(0.02, 0.98, stats_text,
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)

            return []

        # Calculate number of frames
        num_steps = int(self.max_time / self.dt)
        num_frames = num_steps // 5  # 5 steps per frame

        # Create animation
        anim = FuncAnimation(fig, update, frames=num_frames,
                           init_func=init, interval=50, blit=False)

        plt.tight_layout()
        plt.show()

        # Final statistics
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)
        print(f"Total time: {self.t:.1f}")
        print(f"Total rain added: {self.total_rain_added:.1f}")
        print(f"Total infiltrated: {self.total_infiltrated:.1f}")
        print(f"Total drained: {self.total_drained:.1f}")
        print(f"Remaining on surface: {np.sum(self.water):.1f}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='City Rainfall Wave Simulation')
    parser.add_argument('--rainfall', type=str, default='moderate',
                       choices=['drizzle', 'light', 'moderate', 'heavy', 'severe', 'extreme'],
                       help='Rainfall intensity: drizzle(0.03), light(0.08), moderate(0.18), heavy(0.35), severe(0.60), extreme(1.00)')
    parser.add_argument('--duration', type=str, default='standard',
                       choices=['standard', 'storm', 'day', 'week', '10day', 'month'],
                       help='Rain duration: standard(50 units), storm(150 units continuous), day(300 units continuous), week(700 units continuous), 10day(1000 units continuous), month(1500 units continuous)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("CITY RAINFALL WAVE SIMULATION")
    print("="*70)

    # Create landscape
    print("\nCreating city landscape...")
    landscape = CityPotentialLandscape(grid_size_x=200, grid_size_y=150)

    # Create rainfall simulation
    print("\nInitializing rainfall simulation...")
    sim = RainfallWaveSimulation(landscape, rainfall_intensity=args.rainfall, duration=args.duration)

    # Run simulation with animation
    sim.run_simulation()

    print("✓ Simulation complete!")


if __name__ == "__main__":
    main()
