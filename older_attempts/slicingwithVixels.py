import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance

class VoxelObject:
    def __init__(self, x_size, y_size, z_size):
        self.grid = np.zeros((x_size, y_size, z_size), dtype=bool)

    def add_voxel(self, x, y, z):
        self.grid[x, y, z] = True

    def add_filled_cube(self, x_start, y_start, z_start, size):
        for x in range(x_start, x_start + size):
            for y in range(y_start, y_start + size):
                for z in range(z_start, z_start + size):
                    self.add_voxel(x, y, z)

    def find_closest_voxels_to_line(self, line_start, line_end):
        filled_voxels = np.argwhere(self.grid)
        distances = []
        for voxel in filled_voxels:
            point = np.array(voxel)
            dist = distance.euclidean(point, line_start) + distance.euclidean(point, line_end)
            distances.append(dist)

        closest_indices = np.argsort(distances)[:2]
        voxel1, voxel2 = filled_voxels[closest_indices[0]], filled_voxels[closest_indices[1]]
        return voxel1, voxel2

    def slice_object(self, line_start, line_end):
        voxel1, voxel2 = self.find_closest_voxels_to_line(line_start, line_end)
        
        # Create two new voxel objects for the sliced parts
        obj1 = VoxelObject(*self.grid.shape)
        obj2 = VoxelObject(*self.grid.shape)

        # Slicing logic: check each voxel's position relative to the line
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                for z in range(self.grid.shape[2]):
                    if self.grid[x, y, z]:
                        point = np.array([x, y, z])
                        
                        # Calculate the direction of the line
                        line_direction = line_end - line_start
                        # Calculate the vector from the line start to the voxel
                        voxel_vector = point - line_start

                        # Calculate the perpendicular distance to the line
                        cross_product = np.cross(line_direction, voxel_vector)
                        distance_to_line = np.linalg.norm(cross_product) / np.linalg.norm(line_direction)

                        # Check which side of the line the voxel is on
                        if np.dot(line_direction, voxel_vector) > 0:
                            obj2.add_voxel(x, y, z)  # On one side of the line
                        else:
                            obj1.add_voxel(x, y, z)  # On the other side of the line

        return obj1, obj2, voxel1, voxel2

    def display(self, ax, color='b'):
        filled = np.argwhere(self.grid)
        if filled.size > 0:
            x, y, z = filled.T
            ax.scatter(x, y, z, c=color, marker='o', s=100)

# Function to display both objects and the slicing line
def display_both_objects_with_line(obj1, obj2, voxel1, voxel2, line_start, line_end):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Display the first object in red
    obj1.display(ax, color='r')

    # Display the second object in green
    obj2.display(ax, color='g')

    # Plot the slicing line in blue
    ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], [line_start[2], line_end[2]], color='b', linewidth=2, label='Slicing Line')

    # Plot the closest two voxels to the line in yellow
    ax.scatter(*voxel1, c='yellow', s=200, label='Voxel 1')
    ax.scatter(*voxel2, c='yellow', s=200, label='Voxel 2')

    # Hide the axes
    ax.set_axis_off()

    plt.legend()
    plt.show()

# Example usage
def main():
    # Create a 10x10x10 grid
    obj = VoxelObject(10, 10, 10)

    # Add a fully filled cube of size 5x5x5 starting at position (1, 1, 1)
    obj.add_filled_cube(1, 1, 1, 5)

    # Define the slicing line using two points in 3D space within the voxel grid
    line_start = np.array([2, 0, 0])  # Start point of the line within voxel range
    line_end = np.array([2, 10, 10])   # End point of the line within voxel range

    # Slice the object into two parts
    obj1, obj2, voxel1, voxel2 = obj.slice_object(line_start, line_end)

    # Display both objects with the slicing line
    display_both_objects_with_line(obj1, obj2, voxel1, voxel2, line_start, line_end)

if __name__ == "__main__":
    main()
