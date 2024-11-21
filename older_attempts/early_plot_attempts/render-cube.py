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
        self.cube = np.array([[ [Node() for _ in range(z_dim)] for _ in range(y_dim)] for _ in range(x_dim)])

    def plot(self):
        x_dim, y_dim, z_dim = self.dimensions

        # Create a figure for 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a solid cube
        x = np.linspace(0, x_dim, x_dim)
        y = np.linspace(0, y_dim, y_dim)
        z = np.linspace(0, z_dim, z_dim)
        X, Y, Z = np.meshgrid(x, y, z)

        # Define the faces of the cube
        faces = [
            [X[0, :, :], Y[0, :, :], Z[0, :, :]],  # Bottom face
            [X[:, 0, :], Y[:, 0, :], Z[:, 0, :]],  # Left face
            [X[:, :, 0], Y[:, :, 0], Z[:, :, 0]],  # Front face
            [X[-1, :, :], Y[-1, :, :], Z[-1, :, :]],  # Top face
            [X[:, -1, :], Y[:, -1, :], Z[:, -1, :]],  # Right face
            [X[:, :, -1], Y[:, :, -1], Z[:, :, -1]],  # Back face
        ]

        # Plot each face of the cube
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']  # Colors for each face
        for i, face in enumerate(faces):
            ax.plot_surface(face[0], face[1], face[2], color=colors[i], alpha=0.5)

        # Add interactivity with button and hover
        def on_click(event):
            azim, elev = ax.azim, ax.elev
            print("Azimuth:", azim, "Elevation:", elev)

        cid = fig.canvas.mpl_connect('button_release_event', on_click)

        # Create a button for resetting the dissection path
        reset_ax = fig.add_axes([0.7, 0.01, 0.1, 0.05])  # Position of the button
        reset_button = Button(reset_ax, 'Reset Path')

        # Label axes
        ax.set_xlabel('x-Width')
        ax.set_ylabel('y-Depth')
        ax.set_zlabel('z-Height')

        # Set axis limits and ticks
        ax.set_xlim([0, x_dim])
        ax.set_ylim([0, y_dim])
        ax.set_zlim([0, z_dim])
        ax.set_xticks(np.arange(0, x_dim, 1))
        ax.set_yticks(np.arange(0, y_dim, 1))
        ax.set_zticks(np.arange(0, z_dim, 1))

        # Hide the plot background
        fig.patch.set_alpha(0)
        ax.set_facecolor((1, 1, 1, 0))

        # Show the plot
        plt.show()

# Example usage
cube = Cube((9, 5, 5))
cube.plot()
