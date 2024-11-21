import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

class Node:
    def __init__(self, value=1, is_in_dissection_path=False):
        self.value = value
        self.is_in_dissection_path = is_in_dissection_path

class Shape3D:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def plot(self):
        raise NotImplementedError("Subclasses should implement this method!")

class Cube(Shape3D):
    def __init__(self, dimensions):
        super().__init__(dimensions)
        x_dim, y_dim, z_dim = dimensions
        # Create a 3D array of Node objects
        self.cube = np.array([[[Node() for _ in range(z_dim)] for _ in range(y_dim)] for _ in range(x_dim)])

    def plot(self):
        x_dim, y_dim, z_dim = self.dimensions

        # Extract the coordinates of the nodes where value == 1
        x, y, z = np.where(np.array([[[node.value for node in layer] for layer in plane] for plane in self.cube]) == 1)

        # Define the corner points of the cube
        corners = np.array([[0, 0, 0], [0, 0, z_dim-1], [0, y_dim-1, 0], [0, y_dim-1, z_dim-1],
                            [x_dim-1, 0, 0], [x_dim-1, 0, z_dim-1], [x_dim-1, y_dim-1, 0], [x_dim-1, y_dim-1, z_dim-1]])

        # Define the edges (using corner indices)
        edges = [(0, 1), (0, 2), (0, 4),
                 (1, 3), (2, 3), (2, 6), (4, 5),
                 (5, 7), (3, 7), (6, 7), (4, 6),
                 (1, 5)]

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Handle mouse click event
        def on_click(event):
            azim, elev, roll = ax.azim, ax.elev, ax.roll
            print("Azimuth:", azim, "Elevation:", elev, "Roll", roll)

        cid = fig.canvas.mpl_connect('button_release_event', on_click)

        # Create an initial color array for the nodes
        node_colors = np.full_like(x, fill_value='blue', dtype=object)
        outer_node_indices = []  # List to hold indices of outer nodes
        hovered_outer_nodes = set()  # Set to hold currently hovered outer nodes
        dissection_path_coords = []  # List to hold coordinates of dissection path

        # Determine which nodes are outer nodes
        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(z_dim):
                    if (i == 0 or i == x_dim - 1 or
                        j == 0 or j == y_dim - 1 or
                        k == 0 or k == z_dim - 1):
                        outer_node_indices.append((i, j, k))

        # Create scatter plot with clickable points and outlines
        scatter = ax.scatter(x, y, z, c=node_colors, marker='o', picker=True, edgecolor='black', linewidth=1)

        # Plot the edges of the cube
        for edge in edges:
            p1, p2 = corners[edge[0]], corners[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linewidth=2)

        # Define the tooltip
        tooltip = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, color='black')
        tooltip.set_visible(False)  # Initially hidden

        # Connect the hover event
        def on_hover(event):
            if event.inaxes == ax:
                # Get the indices of the points
                cont, ind = scatter.contains(event)
                if cont:
                    # Get the index of the hovered point
                    idx = ind["ind"][0]
                    # Retrieve the corresponding Node
                    node = self.cube[x[idx], y[idx], z[idx]]
                    # Update tooltip text to include is_in_dissection_path
                    azim = ax.azim  # Get current azimuth
                    elev = ax.elev  # Get current elevation
                    tooltip.set_text(
                        f"Node({x[idx]}, {y[idx]}, {z[idx]}), In Dissection Path: {node.is_in_dissection_path}\n"
                        f"Azimuth: {azim}, Elevation: {elev}"
                    )
                    tooltip.set_visible(True)

                    # Change color for outer nodes
                    if True:
                        # Add hovered outer node to the set
                        hovered_outer_nodes.add((x[idx], y[idx], z[idx]))
                        # Change the color of this outer node to red
                        node_colors[idx] = 'red'
                        scatter.set_color(node_colors)  # Update scatter plot color

                        # Set is_in_dissection_path to True for the hovered node
                        self.cube[x[idx], y[idx], z[idx]].is_in_dissection_path = True
                        dissection_path_coords.append((x[idx], y[idx], z[idx]))  # Add to dissection path coordinates

                        # Redraw dissection path line
                        ax.plot(*zip(*dissection_path_coords), color='green', linewidth=2)  # Plot the line

                else:
                    tooltip.set_visible(False)

                plt.draw()

        # Connect the hover event to the on_hover function
        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        # Define the reset function
        def reset_dissection_path(event):
            # Reset the colors of all nodes
            node_colors[:] = 'blue'  # Reset all node colors to blue
            scatter.set_color(node_colors)  # Update scatter plot color

            # Reset is_in_dissection_path for all nodes and clear dissection path coordinates
            for i in range(x_dim):
                for j in range(y_dim):
                    for k in range(z_dim):
                        self.cube[i, j, k].is_in_dissection_path = False
            dissection_path_coords.clear()  # Clear the dissection path coordinates

            plt.draw()

        # Create a button for resetting the dissection path
        reset_ax = fig.add_axes([0.7, 0.01, 0.1, 0.05])  # Position of the button
        reset_button = Button(reset_ax, 'Reset Path')
        reset_button.on_clicked(reset_dissection_path)

        # Create a button to loop through the original 3D array and separate nodes
        loop_ax = fig.add_axes([0.8, 0.01, 0.1, 0.05])  # Position of the new button
        loop_button = Button(loop_ax, 'Separate Nodes')

        # Define the function for the new button
        def separate_nodes(event):
            # Create lists to hold nodes in and not in dissection path
            nodes_in_path = []
            nodes_not_in_path = []

            for i in range(x_dim):
                for j in range(y_dim):
                    for k in range(z_dim):
                        node = self.cube[i, j, k]
                        if node.is_in_dissection_path:
                            nodes_in_path.append((i, j, k))
                        else:
                            nodes_not_in_path.append((i, j, k))

            # Display the results in new 3D plots
            self.plot_nodes_in_new_figures(nodes_in_path, nodes_not_in_path)

        loop_button.on_clicked(separate_nodes)

        # Label axes
        ax.set_xlabel('x-Width')
        ax.set_ylabel('y-Depth')
        ax.set_zlabel('z-Height')

        # Set axis limits and ticks
        ax.set_xlim([0, x_dim - 1])
        ax.set_ylim([0, y_dim - 1])
        ax.set_zlim([0, z_dim - 1])
        ax.set_xticks(np.arange(0, x_dim, 1))
        ax.set_yticks(np.arange(0, y_dim, 1))
        ax.set_zticks(np.arange(0, z_dim, 1))

        # Set the aspect ratio to be equal across all axes
        max_range = np.array([x_dim, y_dim, z_dim]).max()
        ax.set_xlim(0, max_range)
        ax.set_ylim(0, max_range)
        ax.set_zlim(0, max_range)

        plt.show()

    def plot_nodes_in_new_figures(self, nodes_in_path, nodes_not_in_path):
        # Create new 3D figures to display separated nodes
        fig = plt.figure(figsize=(10, 5))
      
        # Create a 3D subplot for nodes in dissection path
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title('Nodes in Dissection Path')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax1.set_zlabel('Z-axis')

        # Plot nodes in the dissection path
        #for node in nodes_in_path:
            #ax1.scatter(node[0], node[1], node[2], c='green', marker='o')
        edges = set()
        for i, p1 in enumerate(nodes_in_path):
            for j, p2 in enumerate(nodes_in_path):
                if i != j:
                    diff = np.abs(np.array(p1) - np.array(p2))
                    # Add edges for adjacent points along each axis (outer edges)
                    if np.sum(diff) == 1:
                        edges.add(tuple(sorted((tuple(p1), tuple(p2)))))
                    # Add diagonal edges for points in the same plane (e.g., XY, XZ, YZ) with equal change along both axes
                    elif (diff[0] == diff[1] and diff[2] == 0) or (diff[0] == diff[2] and diff[1] == 0) or (diff[1] == diff[2] and diff[0] == 0):
                        edges.add(tuple(sorted((tuple(p1), tuple(p2)))))
        for edge in edges:
            x_vals = [edge[0][0], edge[1][0]]
            y_vals = [edge[0][1], edge[1][1]]
            z_vals = [edge[0][2], edge[1][2]]
            ax1.plot(x_vals, y_vals, z_vals, color='k', linewidth=2)
        ax1.set_xlim([-1, 5])
        ax1.set_ylim([-1, 5])
        ax1.set_zlim([-1, 5])

        # Create a 3D subplot for nodes not in dissection path
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title('Nodes Not in Dissection Path')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        ax2.set_zlabel('Z-axis')

        # Plot nodes not in the dissection path
        #for node in nodes_not_in_path:
            #ax2.scatter(node[0], node[1], node[2], c='red', marker='o')
        edges = set()
        for i, p1 in enumerate(nodes_not_in_path):
            for j, p2 in enumerate(nodes_not_in_path):
                if i != j:
                    diff = np.abs(np.array(p1) - np.array(p2))
                    # Add edges for adjacent points along each axis (outer edges)
                    if np.sum(diff) == 1:
                        edges.add(tuple(sorted((tuple(p1), tuple(p2)))))
                    # Add diagonal edges for points in the same plane (e.g., XY, XZ, YZ) with equal change along both axes
                    elif (diff[0] == diff[1] and diff[2] == 0) or (diff[0] == diff[2] and diff[1] == 0) or (diff[1] == diff[2] and diff[0] == 0):
                        edges.add(tuple(sorted((tuple(p1), tuple(p2)))))
        for edge in edges:
            x_vals = [edge[0][0], edge[1][0]]
            y_vals = [edge[0][1], edge[1][1]]
            z_vals = [edge[0][2], edge[1][2]]
            ax2.plot(x_vals, y_vals, z_vals, color='k', linewidth=2)
        ax2.set_xlim([-1, 5])
        ax2.set_ylim([-1, 5])
        ax2.set_zlim([-1, 5])

        plt.tight_layout()
        plt.show()

# Create a cube shape with specified dimensions
cube_shape = Cube(dimensions=(4, 4, 4))
cube_shape.plot()