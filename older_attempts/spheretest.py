import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VoxelObject:
    def __init__(self, x_size, y_size, z_size):
        #a 3D array to represent the object with given dimensions
        self.grid = np.zeros((x_size, y_size, z_size), dtype=bool)
        self.colors = np.empty((x_size, y_size, z_size), dtype=object)

    def add_voxel(self, x, y, z):
        # Mark the voxel at the specified x, y, z coordinate as filled
        self.grid[x, y, z] = True

    def add_filled_cube(self, x_start, y_start, z_start, size):
        # Fill a cube starting from x_start, y_start, z_start with given size
        for x in range(x_start, x_start + size):
            for y in range(y_start, y_start + size):
                for z in range(z_start, z_start + size):
                    self.add_voxel(x, y, z)

    def draw_sphere(self, ax, radius=5, center=(5, 5, 5)):
        # 3D grid for the sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the sphere surface
        ax.plot_surface(x, y, z, color='b', rstride=5, cstride=5, alpha=0.6)

    def display(self):
        # 3D plot for the filled object
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw
        self.draw_sphere(ax)

        
        ax.set_axis_off()

        plt.show()


def main():
   
    obj = VoxelObject(10, 10, 10)
    obj.display()

if __name__ == "__main__":
    main()
