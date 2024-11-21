import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Given list of points
points = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 0), (1, 3, 1), (1, 3, 2), (1, 3, 3), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 0, 3), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 2, 3), (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 0, 3), (3, 1, 0), (3, 1, 1), (3, 1, 2), (3, 1, 3)]


# Convert points to numpy array
points = np.array(points)

# Compute the convex hull to get the outer boundary points
hull = ConvexHull(points)

# Plotting the points and the convex hull outline
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the points
#ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')

# Plot the edges of the convex hull by connecting vertices
for i in range(len(hull.vertices)):
    for j in range(i + 1, len(hull.vertices)):
        ax.plot(
            [points[hull.vertices[i], 0], points[hull.vertices[j], 0]],
            [points[hull.vertices[i], 1], points[hull.vertices[j], 1]],
            [points[hull.vertices[i], 2], points[hull.vertices[j], 2]],
            'k-', linewidth=2
        )

# Setting labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
