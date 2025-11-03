"""
Simple City Potential Landscape Simulation
==========================================

A 2D city on a potential energy landscape with:
- Three regions: Western plateau/slope, Middle flat area with park/hill, Eastern lakebed
- Grid-organized buildings (not random)
- Underground drainage system with 4 main channels
- Visualized as side view (E-W vs Potential) and top view (bird's eye)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from typing import Tuple, List

class CityPotentialLandscape:
    """Creates and visualizes a city on a potential energy landscape"""

    def __init__(self, grid_size_x: int = 200, grid_size_y: int = 150):
        """
        Initialize the landscape

        Args:
            grid_size_x: Grid points in east-west direction
            grid_size_y: Grid points in north-south direction
        """
        self.nx = grid_size_x
        self.ny = grid_size_y

        # Create coordinate grids (normalized 0 to 1)
        self.x = np.linspace(0, 1, self.nx)
        self.y = np.linspace(0, 1, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Initialize potential landscape (surface level)
        self.surface_potential = np.zeros((self.ny, self.nx))

        # Initialize underground drainage level
        self.drainage_potential = np.zeros((self.ny, self.nx))

        # Track building locations
        self.buildings = []  # List of (x_min, x_max, y_min, y_max, height)

        # Track drainage channels
        self.main_channels = []  # List of (y_position, x_start, x_end, depth)
        self.branch_channels = []  # List of (x, y_start, y_end, depth)
        self.surface_openings = []  # List of (x, y)

        # Create the landscape
        self._create_base_terrain()
        self._create_drainage_system()
        self._place_buildings()
        self._add_infiltration_wells()  # Add small wells for water retention

    def _create_base_terrain(self):
        """Create the three-region terrain"""
        print("Creating base terrain...")

        # Region boundaries
        west_end = 0.30  # Western region: 0 to 0.30
        middle_end = 0.70  # Middle region: 0.30 to 0.70
        # East region: 0.70 to 1.0

        for i in range(self.ny):
            for j in range(self.nx):
                x_pos = self.x[j]
                y_pos = self.y[i]

                # REGION 1: WEST - Plateau then slope
                if x_pos < west_end:
                    # Plateau for first third, then slope down
                    plateau_end = west_end * 0.4
                    if x_pos < plateau_end:
                        self.surface_potential[i, j] = 50.0  # High plateau
                    else:
                        # Linear slope from 50 to 20
                        slope_progress = (x_pos - plateau_end) / (west_end - plateau_end)
                        self.surface_potential[i, j] = 50.0 - 30.0 * slope_progress

                # REGION 2: MIDDLE - Flat with park (north) and hill (south)
                elif x_pos < middle_end:
                    # Check if in northern part (park area - flat and empty)
                    if y_pos > 0.6:
                        self.surface_potential[i, j] = 20.0  # Flat park level
                    # Southern part has a small hill
                    elif y_pos < 0.4:
                        # Small hill in the south-middle area
                        hill_center_x = (west_end + middle_end) / 2
                        hill_center_y = 0.25
                        dist_to_hill = np.sqrt((x_pos - hill_center_x)**2 * 4 + (y_pos - hill_center_y)**2 * 4)
                        if dist_to_hill < 0.15:
                            # Gaussian hill
                            hill_height = 15.0 * np.exp(-dist_to_hill**2 / 0.01)
                            self.surface_potential[i, j] = 20.0 + hill_height
                        else:
                            self.surface_potential[i, j] = 20.0  # Flat
                    else:
                        self.surface_potential[i, j] = 20.0  # Flat center

                # REGION 3: EAST - Deep lakebed (shore)
                else:
                    # STRAIGHT DROP to lakebed (vertical cliff)
                    self.surface_potential[i, j] = -10.0  # Immediate drop to lakebed level

        print("  ✓ Base terrain created")
        print(f"    - West plateau: {0.0:.1%} to {west_end:.1%}")
        print(f"    - Middle flat: {west_end:.1%} to {middle_end:.1%}")
        print(f"    - East lakebed: {middle_end:.1%} to {1.0:.1%}")

    def _create_drainage_system(self):
        """Create underground drainage with 4 main channels and branches"""
        print("Creating drainage system...")

        # 4 main channels running east-west
        # Positioned at y = 0.2, 0.4, 0.6, 0.8
        channel_y_positions = [0.2, 0.4, 0.6, 0.8]
        channel_depth = 15.0  # Just below surface (surface is ~20.0, so this creates ~5 unit gap)
        channel_width = 0.03  # Width in normalized coords (wider for visibility)
        channel_height = 3.0  # Vertical extent of hollow channel

        # Main channels only in west and middle regions (0 to 0.70)
        channel_length = 0.70

        for y_pos in channel_y_positions:
            y_idx = int(y_pos * self.ny)
            y_start = max(0, y_idx - int(channel_width * self.ny / 2))
            y_end = min(self.ny, y_idx + int(channel_width * self.ny / 2))

            # Create channel from west to 70% east
            x_end = int(channel_length * self.nx)
            for j in range(x_end):
                for i in range(y_start, y_end):
                    self.drainage_potential[i, j] = channel_depth

            self.main_channels.append((y_pos, 0.0, channel_length, channel_depth))

        # Create vertical branch channels connecting to surface
        # Spaced every ~8% along x-axis
        branch_spacing = 0.08
        branch_x_positions = np.arange(0.05, channel_length, branch_spacing)
        branch_opening_width = 0.015  # Width of surface opening

        for x_pos in branch_x_positions:
            x_idx = int(x_pos * self.nx)

            # For each main channel, create a branch upward
            for y_pos in channel_y_positions:
                y_idx = int(y_pos * self.ny)

                # Vertical branch from channel to surface (create hollow passage)
                branch_width_cells = max(3, int(branch_opening_width * self.nx))
                x_start = max(0, x_idx - branch_width_cells // 2)
                x_end = min(self.nx, x_idx + branch_width_cells // 2)

                # Create vertical opening from surface down to channel
                for j in range(x_start, x_end):
                    for i in range(y_idx - 3, y_idx + 4):  # Wider vertical extent
                        if 0 <= i < self.ny:
                            # Set to channel depth to show hollow connection
                            self.drainage_potential[i, j] = channel_depth

                # CRITICAL: Cut through the surface to create actual openings!
                # This creates a hole from surface level down to drainage level
                for j in range(x_start, x_end):
                    for i in range(y_idx - 3, y_idx + 4):
                        if 0 <= i < self.ny:
                            # Create opening by setting surface to drainage level
                            # This makes the hole visible in cross-section
                            self.surface_potential[i, j] = channel_depth

                # Mark surface opening
                self.surface_openings.append((x_pos, y_pos))
                self.branch_channels.append((x_pos, y_pos - 0.02, y_pos + 0.02, channel_depth))

        print(f"  ✓ Drainage system created")
        print(f"    - {len(self.main_channels)} main channels")
        print(f"    - {len(self.surface_openings)} surface openings")

    def _place_buildings(self):
        """Place buildings in grid pattern with roads and sidewalks"""
        print("Placing buildings...")

        # Building parameters
        small_building_height = 25.0  # Small houses
        tall_building_height = 80.0   # Skyscrapers
        building_size = 0.035  # Footprint size (WIDER buildings)
        grid_spacing = 0.10    # Spacing between buildings (MORE space for roads)

        # Road and sidewalk parameters
        road_depression = -3.0  # Roads are depressed below surface
        sidewalk_elevation = 2.0  # Sidewalks slightly elevated above road

        building_count = 0

        # REGION 1: WEST - Small buildings on plateau
        for x in np.arange(0.05, 0.25, grid_spacing):
            for y in np.arange(0.15, 0.85, grid_spacing):
                # Skip if in drainage opening
                if self._near_drainage_opening(x, y, radius=0.015):
                    continue
                self._add_building(x, y, building_size, small_building_height)
                building_count += 1

        # REGION 2: MIDDLE - Buildings with special zones
        west_end = 0.30
        middle_end = 0.70

        for x in np.arange(west_end + 0.03, middle_end - 0.03, grid_spacing):
            for y in np.arange(0.05, 0.95, grid_spacing):
                # Skip park area (north center) except borders
                if 0.6 < y < 0.9:
                    # Only place on borders (2 rows on each side)
                    if not (x < west_end + 0.08 or x > middle_end - 0.08):
                        continue

                # Skip hill area (small buildings on hill handled separately)
                if y < 0.4:
                    hill_center_x = (west_end + middle_end) / 2
                    hill_center_y = 0.25
                    dist_to_hill = np.sqrt((x - hill_center_x)**2 * 4 + (y - hill_center_y)**2 * 4)
                    if dist_to_hill < 0.12:
                        # On hill - add small building (no roads on hill)
                        if np.random.random() > 0.5:  # Sparse on hill
                            self._add_building(x, y, building_size * 0.8, small_building_height * 0.7,
                                             add_roads=False)
                            building_count += 1
                        continue

                # Skip drainage openings
                if self._near_drainage_opening(x, y, radius=0.015):
                    continue

                # Near shore (x > 0.60): TALL skyscrapers
                if x > 0.60:
                    self._add_building(x, y, building_size * 1.2, tall_building_height)
                else:
                    self._add_building(x, y, building_size, small_building_height)
                building_count += 1

        print(f"  ✓ {building_count} buildings placed in grid pattern")
        print(f"    - Small buildings: ~{building_count - 20}")
        print(f"    - Tall skyscrapers: ~20 (near shore)")

    def _near_drainage_opening(self, x: float, y: float, radius: float = 0.015) -> bool:
        """Check if position is near a drainage opening"""
        for ox, oy in self.surface_openings:
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < radius:
                return True
        return False

    def _add_building(self, x: float, y: float, size: float, height: float,
                      add_roads: bool = True, road_width: float = 0.04):
        """Add a building to the landscape with roads and sidewalks"""
        x_idx = int(x * self.nx)
        y_idx = int(y * self.ny)

        half_size = int(size * self.nx / 2)
        road_depression = -3.0
        sidewalk_elevation = 2.0
        sidewalk_width = int(0.008 * self.nx)  # Sidewalk width in pixels

        x_min = max(0, x_idx - half_size)
        x_max = min(self.nx, x_idx + half_size)
        y_min = max(0, y_idx - half_size)
        y_max = min(self.ny, y_idx + half_size)

        if add_roads:
            # Create road zone around building (depressed)
            road_half_width = int(road_width * self.nx / 2)
            road_x_min = max(0, x_idx - half_size - road_half_width)
            road_x_max = min(self.nx, x_idx + half_size + road_half_width)
            road_y_min = max(0, y_idx - half_size - road_half_width)
            road_y_max = min(self.ny, y_idx + half_size + road_half_width)

            # Apply road depression
            for i in range(road_y_min, road_y_max):
                for j in range(road_x_min, road_x_max):
                    # Only modify if not already a building
                    if self.surface_potential[i, j] < 30:  # Not a building
                        base_height = self.surface_potential[i, j]
                        self.surface_potential[i, j] = base_height + road_depression

            # Create sidewalks (elevated margin between road and building)
            sidewalk_inner_x_min = x_min - sidewalk_width
            sidewalk_inner_x_max = x_max + sidewalk_width
            sidewalk_inner_y_min = y_min - sidewalk_width
            sidewalk_inner_y_max = y_max + sidewalk_width

            for i in range(max(0, sidewalk_inner_y_min), min(self.ny, sidewalk_inner_y_max)):
                for j in range(max(0, sidewalk_inner_x_min), min(self.nx, sidewalk_inner_x_max)):
                    # Sidewalk zone: between road and building
                    if not (y_min <= i < y_max and x_min <= j < x_max):  # Not the building itself
                        base_height = self.surface_potential[i, j]
                        # Elevate above road
                        self.surface_potential[i, j] = base_height - road_depression + sidewalk_elevation

        # Raise potential at building location (building sits on top)
        for i in range(y_min, y_max):
            for j in range(x_min, x_max):
                # Building sits on top of terrain
                base_height = self.surface_potential[i, j]
                # Remove any road depression first, then add building
                if base_height < 0:
                    base_height = 20.0  # Reset to flat surface level
                self.surface_potential[i, j] = base_height + height

        self.buildings.append((x_min/self.nx, x_max/self.nx, y_min/self.ny, y_max/self.ny, height))

    def _add_infiltration_wells(self):
        """Add small wells across entire surface for water infiltration/retention"""
        print("Adding infiltration wells...")

        # Region boundaries
        west_end = 0.30
        middle_end = 0.70

        # Well spacing (MUCH denser - more micro wells everywhere!)
        well_spacing = 2  # pixels between wells (VERY dense - was 3)
        well_count = 0

        for i in range(0, self.ny, well_spacing):
            for j in range(0, self.nx, well_spacing):
                x_pos = self.x[j]
                y_pos = self.y[i]

                # Determine well depth based on location
                well_depth = 0.0

                # Skip buildings (buildings don't have infiltration wells)
                if self.surface_potential[i, j] > 40:  # This is a building
                    continue

                # PARK AREA (north of middle region): DEEPEST wells
                if west_end < x_pos < middle_end and y_pos > 0.6:
                    well_depth = -2.5  # MUCH deeper infiltration (best soil)

                # SLOPED AREA (west region): Create STAIR-STEP effect
                elif x_pos < west_end:
                    # Check if on slope - create stepped terraces
                    if self.surface_potential[i, j] < 40 and self.surface_potential[i, j] > 25:
                        # Create visible steps by varying depth with position
                        # More pronounced steps for staircase effect
                        well_depth = -1.2 - 0.3 * np.sin(j * 0.5)  # Stepped pattern
                    else:
                        well_depth = -1.5  # Flat areas on plateau (deeper)

                # REGULAR ROADS/FLAT SURFACES: DEEP micro wells
                else:
                    well_depth = -1.8  # Deeper standard infiltration

                # Apply small well depression (slightly larger radius)
                well_radius = 1  # Keep small but apply more strongly
                for di in range(-well_radius, well_radius + 1):
                    for dj in range(-well_radius, well_radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.ny and 0 <= nj < self.nx:
                            # Don't modify buildings
                            if self.surface_potential[ni, nj] < 40:
                                # Add depression (STRONGER effect - was 0.5, now 0.8)
                                self.surface_potential[ni, nj] += well_depth * 0.8
                                # Also add to drainage if it exists there
                                if self.drainage_potential[ni, nj] > 0:
                                    self.drainage_potential[ni, nj] += well_depth * 0.8

                well_count += 1

        print(f"  ✓ {well_count} infiltration wells created (DENSE micro-topography)")
        print(f"    - Park areas: Very deep wells (-2.5 depression)")
        print(f"    - Flat surfaces: Deep wells (-1.8 depression)")
        print(f"    - Sloped areas: Stepped wells (-1.2 to -1.5, stair pattern)")
        print(f"    - Total coverage: Every {well_spacing} pixels")

    def visualize(self):
        """Create both side view and top view visualizations"""
        fig = plt.figure(figsize=(16, 12))

        # ===== PLOT 1: SIDE VIEW (East-West vs Potential) =====
        ax1 = fig.add_subplot(2, 2, 1)

        # Average potential across north-south to get side profile
        side_profile = np.mean(self.surface_potential, axis=0)
        drainage_profile = np.mean(self.drainage_potential, axis=0)

        # Show ground level below drainage (solid earth)
        ground_level = np.full_like(self.x, 0.0)

        # Fill solid ground (below drainage)
        ax1.fill_between(self.x, ground_level, drainage_profile,
                         color='saddlebrown', alpha=0.8, label='Solid Ground')

        # Show hollow gap between drainage and surface (this is the channel space)
        # Only where drainage exists (not zeros)
        has_drainage = drainage_profile > 1.0

        # Fill from drainage to surface only where there's structure
        ax1.fill_between(self.x, drainage_profile, side_profile,
                         where=~has_drainage,
                         color='sienna', alpha=0.7, label='Filled Ground')

        # Drainage channels as distinct layer
        ax1.plot(self.x, drainage_profile, 'b-', linewidth=2, label='Drainage Channel Ceiling', alpha=0.8)

        # Surface profile
        ax1.plot(self.x, side_profile, 'k-', linewidth=2, label='Surface Profile')

        # Mark regions
        ax1.axvline(0.30, color='red', linestyle='--', alpha=0.5, label='Region Boundaries')
        ax1.axvline(0.70, color='red', linestyle='--', alpha=0.5)

        ax1.set_xlabel('Position (West ← → East)', fontsize=12)
        ax1.set_ylabel('Potential Energy', fontsize=12)
        ax1.set_title('SIDE VIEW: East-West Profile', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, 1)

        # ===== PLOT 2: TOP VIEW (Bird's Eye - 2D Map) =====
        ax2 = fig.add_subplot(2, 2, 2)

        # Show surface potential as color map
        im = ax2.imshow(self.surface_potential, extent=[0, 1, 0, 1],
                       origin='lower', cmap='terrain', aspect='auto')
        plt.colorbar(im, ax=ax2, label='Potential Energy')

        # Mark drainage openings
        if self.surface_openings:
            openings_x = [o[0] for o in self.surface_openings]
            openings_y = [o[1] for o in self.surface_openings]
            ax2.scatter(openings_x, openings_y, c='blue', s=20, marker='o',
                       edgecolors='white', linewidths=0.5, label='Drainage Openings')

        # Mark regions
        ax2.axvline(0.30, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax2.axvline(0.70, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax2.axhline(0.6, color='green', linestyle='--', alpha=0.3, linewidth=1)  # Park boundary

        ax2.set_xlabel('Position (West ← → East)', fontsize=12)
        ax2.set_ylabel('Position (South ← → North)', fontsize=12)
        ax2.set_title('TOP VIEW: Bird\'s Eye (Buildings as High Potential)', fontsize=14, fontweight='bold')
        ax2.legend()

        # ===== PLOT 3: DRAINAGE SYSTEM VIEW =====
        ax3 = fig.add_subplot(2, 2, 3)

        # Show drainage potential
        drainage_vis = self.drainage_potential.copy()
        drainage_vis[drainage_vis == 0] = np.nan  # Only show actual drainage

        im3 = ax3.imshow(drainage_vis, extent=[0, 1, 0, 1],
                        origin='lower', cmap='Blues_r', aspect='auto')
        plt.colorbar(im3, ax=ax3, label='Drainage Depth')

        # Overlay main channel lines
        for y_pos, x_start, x_end, depth in self.main_channels:
            ax3.plot([x_start, x_end], [y_pos, y_pos], 'r-', linewidth=3, alpha=0.7)

        # Mark surface openings
        if self.surface_openings:
            openings_x = [o[0] for o in self.surface_openings]
            openings_y = [o[1] for o in self.surface_openings]
            ax3.scatter(openings_x, openings_y, c='lime', s=30, marker='s',
                       edgecolors='black', linewidths=1, label='Surface Inlets')

        ax3.set_xlabel('Position (West ← → East)', fontsize=12)
        ax3.set_ylabel('Position (South ← → North)', fontsize=12)
        ax3.set_title('DRAINAGE SYSTEM: 4 Channels with Branches', fontsize=14, fontweight='bold')
        ax3.legend()

        # ===== PLOT 4: 3D PERSPECTIVE =====
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')

        # Subsample for performance
        skip = 3
        X_sub = self.X[::skip, ::skip]
        Y_sub = self.Y[::skip, ::skip]
        Z_sub = self.surface_potential[::skip, ::skip]

        surf = ax4.plot_surface(X_sub, Y_sub, Z_sub, cmap='terrain',
                               linewidth=0, antialiased=True, alpha=0.8)

        ax4.set_xlabel('East-West')
        ax4.set_ylabel('North-South')
        ax4.set_zlabel('Potential Energy')
        ax4.set_title('3D VIEW: Potential Landscape', fontsize=14, fontweight='bold')
        ax4.view_init(elev=25, azim=45)

        plt.tight_layout()
        plt.savefig('city_potential_landscape.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved as 'city_potential_landscape.png'")
        plt.show()

    def print_summary(self):
        """Print simulation summary"""
        print("\n" + "="*70)
        print("CITY POTENTIAL LANDSCAPE SIMULATION - SUMMARY")
        print("="*70)
        print(f"\nGrid Size: {self.nx} × {self.ny} points")
        print(f"Total Buildings: {len(self.buildings)}")
        print(f"Drainage Channels: {len(self.main_channels)} main + {len(self.surface_openings)} openings")
        print(f"\nPotential Range:")
        print(f"  Maximum (tallest building): {np.max(self.surface_potential):.1f}")
        print(f"  Minimum (lakebed): {np.min(self.surface_potential):.1f}")
        print(f"  Drainage depth: {np.min(self.drainage_potential):.1f}")
        print("\nRegions:")
        print("  West (0-30%): Plateau and slope")
        print("  Middle (30-70%): Flat city with park (north) and hill (south)")
        print("  East (70-100%): Lakebed (shore)")
        print("="*70 + "\n")


def main():
    """Run the simulation"""
    print("\n" + "="*70)
    print("SIMPLE CITY POTENTIAL LANDSCAPE SIMULATION")
    print("="*70 + "\n")

    # Create landscape
    landscape = CityPotentialLandscape(grid_size_x=200, grid_size_y=150)

    # Print summary
    landscape.print_summary()

    # Visualize
    print("Creating visualizations...")
    landscape.visualize()

    print("\n✓ Simulation complete!")


if __name__ == "__main__":
    main()
