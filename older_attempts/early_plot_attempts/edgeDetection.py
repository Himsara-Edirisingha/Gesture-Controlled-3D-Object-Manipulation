import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Given list of points
points =[(2, 3, 0), (2, 3, 1), (2, 3, 2), (2, 3, 3), (3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3), (3, 3, 0), (3, 3, 1), (3, 3, 2), (3, 3, 3)]

# Convert points to numpy array
points = np.array(points)

# Plotting the points and outlining the edges
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')

# Detecting essential edges (outer and diagonal)
edges = set()
for i, p1 in enumerate(points):
    for j, p2 in enumerate(points):
        if i != j:
            diff = np.abs(p1 - p2)
            # Add edges for adjacent points along each axis (outer edges)
            if np.sum(diff) == 1:
                edges.add(tuple(sorted((tuple(p1), tuple(p2)))))
            # Add diagonal edges for points in the same plane (e.g., XY, XZ, YZ) with equal change along both axes
            elif (diff[0] == diff[1] and diff[2] == 0) or (diff[0] == diff[2] and diff[1] == 0) or (diff[1] == diff[2] and diff[0] == 0):
                edges.add(tuple(sorted((tuple(p1), tuple(p2)))))

# Plot the edges
for edge in edges:
    x_vals = [edge[0][0], edge[1][0]]
    y_vals = [edge[0][1], edge[1][1]]
    z_vals = [edge[0][2], edge[1][2]]
    ax.plot(x_vals, y_vals, z_vals, color='k', linewidth=2)

# Setting labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
