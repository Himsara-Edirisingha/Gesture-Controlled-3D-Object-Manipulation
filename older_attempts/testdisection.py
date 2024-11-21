import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive 3D Node Cube with Dissection")

class Node:
    def __init__(self, position, size=15, color=(0, 0, 0)):
        self.position = position  # Position as a numpy array
        self.size = size          # Size of the node
        self.default_color = color  # Store default color
        self.isOnDissectionPath = False  # Dissection path indicator

    def draw(self, screen, zoom, offset_x, offset_y, rotation_matrices):
        # Apply rotation and projection to determine the 2D screen position
        rotated_position = self.position.dot(rotation_matrices[0]).dot(rotation_matrices[1]).dot(rotation_matrices[2])
        projected_position = rotated_position[:2] * zoom

        # Calculate screen coordinates with offsets
        screen_x = int(projected_position[0] + offset_x)
        screen_y = int(projected_position[1] + offset_y)

        # Set color based on dissection path state
        draw_color = (255, 0, 0) if self.isOnDissectionPath else self.default_color
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
                    nodes.append(Node(position, color=(0, 255, 0)))  # Default color green
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

    def dissect(self):
        # Split nodes into two groups based on whether they are on the dissection path
        group1 = [node for node in self.nodes if node.isOnDissectionPath]
        group2 = [node for node in self.nodes if not node.isOnDissectionPath]
        return group1, group2

    def draw(self, screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, mouse_x, mouse_y):
        # Draw all nodes and check for hover when in dissection mode
        for node in self.nodes:
            screen_x, screen_y, size = node.draw(screen, zoom, offset_x, offset_y, rotation_matrices)

            # Check for mouse hover only when in dissection mode
            if dissection_mode:
                if node.check_hover(mouse_x, mouse_y, screen_x, screen_y):
                    node.isOnDissectionPath = True  # Set the node to be on dissection path
        
        # Draw lines connecting the corners
        self.draw_lines(screen, zoom, offset_x, offset_y, rotation_matrices)

    def draw_lines(self, screen, zoom, offset_x, offset_y, rotation_matrices):
        # Draw lines between corners
        for i in range(len(self.corners)):
            for j in range(i + 1, len(self.corners)):
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
                pygame.draw.line(screen, (255, 255, 255), (screen_x_a, screen_y_a), (screen_x_b, screen_y_b), 2)

# Create a cube object with separate width, height, and length dimensions
cube_width, cube_height, cube_length = 10, 10, 10
cube = Cube(cube_width, cube_height, cube_length)

# Rotation angles, zoom level, and control flags
rotation_x, rotation_y, rotation_z = 0, 0, 0
zoom = 20  # Scale factor for visualization
offset_x, offset_y = width // 2, height // 2
dragging = False  # Track dragging state
dissection_mode = False  # Dissection mode state
new_objects = []  # List to hold new objects after dissection

dissected = False  # Flag to track if dissection has been performed

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
            elif event.key == pygame.K_RETURN and dissection_mode:
                # Perform dissection when Enter key is pressed in dissection mode
                group1, group2 = cube.dissect()
                new_objects = [group1, group2]
                dissected = True
                print(f"Group 1 has {len(group1)} nodes, Group 2 has {len(group2)} nodes.")
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
    screen.fill((0, 0, 0))
    
    # Draw the original cube or the new separated objects after dissection
    if dissected:
        # Draw group 1 on the left side of the screen
        for node in new_objects[0]:
            node.draw(screen, zoom, offset_x - width // 4, offset_y, rotation_matrices)
        
        # Draw group 2 on the right side of the screen
        for node in new_objects[1]:
            node.draw(screen, zoom, offset_x + width // 4, offset_y, rotation_matrices)
    else:
        cube.draw(screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, mouse_x, mouse_y)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
