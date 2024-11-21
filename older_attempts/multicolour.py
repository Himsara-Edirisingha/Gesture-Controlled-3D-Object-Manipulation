import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

class VoxelObject:
    def __init__(self, x_size, y_size, z_size):
        # Create a 3D array to represent the object with given dimensions
        self.grid = np.zeros((x_size, y_size, z_size), dtype=bool)
        self.colors = np.empty((x_size, y_size, z_size), dtype=object)

    def add_voxel(self, x, y, z, color='blue'):
        # Mark the voxel at the specified (x, y, z) coordinate as filled and set its color
        self.grid[x, y, z] = True
        self.colors[x, y, z] = color

    def add_sphere(self, center, radius):
        # Fill a sphere centered at (cx, cy, cz) with a given radius
        cx, cy, cz = center
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                for z in range(self.grid.shape[2]):
                    # Check if the voxel (x, y, z) is inside the sphere using the distance formula
                    if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2:
                        # Assign a random color to the voxel inside the sphere
                        color = np.random.choice(list(mcolors.CSS4_COLORS.values()))
                        self.add_voxel(x, y, z, color)

    def display(self):
        # Create a 3D plot for the filled voxels
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Use `voxels` to display cubes with different colors
        ax.voxels(self.grid, facecolors=self.colors, edgecolors='k')

        # Hide the axes
        ax.set_axis_off()

        plt.show()

# Example usage
def main():
    # Create a 20x20x20 grid
    obj = VoxelObject(20, 20, 20)

    # Add a sphere of radius 8 centered at position (10, 10, 10)
    obj.add_sphere(center=(10, 10, 10), radius=8)

    # Display the 3D object
    obj.display()

if __name__ == "__main__":
    main()
