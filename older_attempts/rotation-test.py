import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 600,500
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive 3D Node Cube")

class Node:
    def __init__(self, position, size=10, color=(0, 0, 0, 255)):
        self.position = position  # Position as a numpy array
        self.size = size          # Size of the node
        self.default_color = color  # Store default color
        self.isOnDissectionPath = False  # Dissection path indicator
        self.dissection_color = None  # Color for dissection highlighting

    def draw(self, screen, zoom, offset_x, offset_y, rotation_matrices):
        # Apply rotation and projection to determine the 2D screen position
        rotated_position = self.position.dot(rotation_matrices[0]).dot(rotation_matrices[1]).dot(rotation_matrices[2])
        projected_position = rotated_position[:2] * zoom

        # Calculate screen coordinates with offsets
        screen_x = int(projected_position[0] + offset_x)
        screen_y = int(projected_position[1] + offset_y)

        # Set color based on dissection path state
        if self.isOnDissectionPath and self.dissection_color:
            draw_color = self.dissection_color
        else:
            draw_color = self.default_color
        pygame.draw.rect(screen, draw_color, (screen_x, screen_y, self.size, self.size))

        # Return screen coordinates and size for hover detection
        return (screen_x, screen_y, self.size)

    def check_hover(self, mouse_x, mouse_y, screen_x, screen_y):
        # Check if the mouse is within the node's rectangle on the screen
        return screen_x <= mouse_x <= screen_x + self.size and screen_y <= mouse_y <= screen_y + self.size

class Cube:
    def __init__(self, width, height, length):
        self.width = width
        self.height = height
        self.length = length
        self.nodes = self.create_nodes()
        self.corners = self.find_corners()

    def create_nodes(self):
        nodes = []
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.length):
                    # Adjust positions based on width, height, and length
                    position = np.array([x - self.width / 2, y - self.height / 2, z - self.length / 2])
                    nodes.append(Node(position, color=(255, 255, 255, 255)))  # Default color white
        return nodes

    def find_corners(self):
        # Find the minimum and maximum values along each axis
        min_x, min_y, min_z = -self.width / 2, -self.height / 2, -self.length / 2
        max_x, max_y, max_z = self.width / 2 - 1, self.height / 2 - 1, self.length / 2 - 1

        # Identify nodes that are at the corners
        corners = []
        for node in self.nodes:
            x, y, z = node.position
            if (x, y, z) in [
                (min_x, min_y, min_z), (min_x, min_y, max_z),
                (min_x, max_y, min_z), (min_x, max_y, max_z),
                (max_x, min_y, min_z), (max_x, min_y, max_z),
                (max_x, max_y, min_z), (max_x, max_y, max_z)
            ]:
                corners.append(node)
        return corners

    def draw(self, screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, mouse_x, mouse_y):
        # Draw all nodes and check for hover when in dissection mode
        for node in self.nodes:
            screen_x, screen_y, size = node.draw(screen, zoom, offset_x, offset_y, rotation_matrices)

            # Check for mouse hover only when in dissection mode
            if dissection_mode:
                if node.check_hover(mouse_x, mouse_y, screen_x, screen_y):
                    node.isOnDissectionPath = True  # Set the node to be on dissection path
                else:
                    node.isOnDissectionPath = False  # Reset if not hovered
        
        # Draw lines connecting the corners
        self.draw_lines(screen, zoom, offset_x, offset_y, rotation_matrices)

    def draw_lines(self, screen, zoom, offset_x, offset_y, rotation_matrices):
        # Define the pairs of corners that form the edges of a cube
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Edges along the bottom face
            (4, 5), (5, 7), (7, 6), (6, 4),  # Edges along the top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges connecting top and bottom faces
        ]

        # Draw lines between the defined corner pairs
        for edge in edges:
            i, j = edge
            corner_a = self.corners[i].position.dot(rotation_matrices[0]).dot(rotation_matrices[1]).dot(rotation_matrices[2])
            corner_b = self.corners[j].position.dot(rotation_matrices[0]).dot(rotation_matrices[1]).dot(rotation_matrices[2])

            # Project the corners to screen
            projected_a = corner_a[:2] * zoom
            projected_b = corner_b[:2] * zoom

            # Calculate screen coordinates with offsets
            screen_x_a = int(projected_a[0] + offset_x)
            screen_y_a = int(projected_a[1] + offset_y)
            screen_x_b = int(projected_b[0] + offset_x)
            screen_y_b = int(projected_b[1] + offset_y)

            # Draw the line
            pygame.draw.line(screen, (0, 0, 0), (screen_x_a, screen_y_a), (screen_x_b, screen_y_b), 2)

    def dissect(self, cutting_panes):
        # Dissect the nodes that are intersecting with any of the cutting panes
        for node in self.nodes:
            if abs(node.position[0] - cutting_panes['x'].position) < 0.6:
                node.isOnDissectionPath = True
                node.dissection_color = (255, 0, 0, 128)  # Red for X-axis dissection
            elif abs(node.position[1] - cutting_panes['y'].position) < 0.6:
                node.isOnDissectionPath = True
                node.dissection_color = (0, 255, 0, 128)  # Green for Y-axis dissection
            elif abs(node.position[2] - cutting_panes['z'].position) < 0.6:
                node.isOnDissectionPath = True
                node.dissection_color = (0, 0, 255, 128)  # Blue for Z-axis dissection
            else:
                node.isOnDissectionPath = False
                node.dissection_color = None

# Create a cube object with separate width, height, and length dimensions
cube_width, cube_height, cube_length = 20, 15, 20
cube = Cube(cube_width, cube_height, cube_length)

# Rotation angles, zoom level, and control flags
rotation_x, rotation_y, rotation_z = 0, 0, 0
zoom = 20  # Scale factor for visualization
offset_x, offset_y = width // 2, height // 2
dragging = False  # Track dragging state
dissection_mode = False  # Dissection mode state

# Game loop
running = True
while running:
    mouse_x, mouse_y = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                dissection_mode = not dissection_mode  # Toggle dissection mode
            elif event.key == pygame.K_w:  # Rotate up around X-axis
                rotation_x += 0.1
            elif event.key == pygame.K_s:  # Rotate down around X-axis
                rotation_x -= 0.1
            elif event.key == pygame.K_a:  # Rotate left around Y-axis
                rotation_y -= 0.1
            elif event.key == pygame.K_d:  # Rotate right around Y-axis
                rotation_y += 0.1
            elif event.key == pygame.K_q:  # Rotate counterclockwise around Z-axis
                rotation_z += 0.1
            elif event.key == pygame.K_e:  # Rotate clockwise around Z-axis
                rotation_z -= 0.1
            elif event.key == pygame.K_UP:
                zoom += 2  # Zoom in
            elif event.key == pygame.K_DOWN:
                zoom = max(5, zoom - 2)  # Zoom out with a minimum limit
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Start dragging when mouse is pressed
                dragging = True
                mouse_x, mouse_y = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Stop dragging when mouse button is released
                dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging:  # Adjust position while dragging
                dx, dy = event.rel
                offset_x += dx
                offset_y += dy

    # Rotation matrices
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_x), -np.sin(rotation_x)],
        [0, np.sin(rotation_x), np.cos(rotation_x)]
    ])
    rotation_matrix_y = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    rotation_matrix_z = np.array([
        [np.cos(rotation_z), -np.sin(rotation_z), 0],
        [np.sin(rotation_z), np.cos(rotation_z), 0],
        [0, 0, 1]
    ])
    rotation_matrices = [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]

    # Clear the screen
    screen.fill((255, 255, 255))

    # Dissect the cube if dissection mode is active
    if dissection_mode:
        cube.dissect({})

    # Draw the cube nodes and lines
    cube.draw(screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, mouse_x, mouse_y)

    # Display rotation values
    font = pygame.font.Font(None, 24)
    rotation_text_x = font.render(f"Rotation X: {np.degrees(rotation_x):.2f}°", True, (0, 0, 0))
    rotation_text_y = font.render(f"Rotation Y: {np.degrees(rotation_y):.2f}°", True, (0, 0, 0))
    rotation_text_z = font.render(f"Rotation Z: {np.degrees(rotation_z):.2f}°", True, (0, 0, 0))

    screen.blit(rotation_text_x, (10, 10))
    screen.blit(rotation_text_y, (10, 40))
    screen.blit(rotation_text_z, (10, 70))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()