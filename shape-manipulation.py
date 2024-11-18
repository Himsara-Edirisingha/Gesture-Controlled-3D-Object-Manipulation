import pygame
import numpy as np
from scipy.spatial import ConvexHull

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive Node Cube with Enhanced Rendering")

# Lighting configuration
light_positions = [
    np.array([0, -500, 0]),  # Light from front
    np.array([0, 500, 0]),   # Light from behind
    np.array([-500, 500, 500])  # Light from top-left
]

class Node:
    def __init__(self, position, size=5, color=(128, 128, 128)):
        self.position = position  # Position as a numpy array
        self.size = size          # Size of the node
        self.default_color = color  # Store default color
        self.isOnDissectionPath = False  # Dissection path indicator

    def calculate_lighting(self, rotation_matrices):
        rotated_position = self.position.dot(rotation_matrices[0]).dot(rotation_matrices[1]).dot(rotation_matrices[2])
        total_intensity = 0

        for light_position in light_positions:
            light_direction = light_position - rotated_position
            light_direction = light_direction / np.linalg.norm(light_direction)
            normal = rotated_position / np.linalg.norm(rotated_position)
            dot_product = np.dot(normal, light_direction)
            intensity = max(0, min(1, dot_product))
            total_intensity += intensity * 1.5

        total_intensity = min(1, total_intensity / len(light_positions))
        base_color = np.array(self.default_color)
        lit_color = (base_color * total_intensity).astype(int)
        return tuple(lit_color)

    def draw(self, screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, dissected):
        rotated_position = self.position.dot(rotation_matrices[0]).dot(rotation_matrices[1]).dot(rotation_matrices[2])
        projected_position = rotated_position[:2] * zoom

        screen_x = int(projected_position[0] + offset_x)
        screen_y = int(projected_position[1] + offset_y)

        draw_color = self.calculate_lighting(rotation_matrices) if dissected else ((255, 0, 0) if self.isOnDissectionPath and dissection_mode else self.calculate_lighting(rotation_matrices))
        pygame.draw.circle(screen, draw_color, (screen_x, screen_y), self.size, 0)
        return (screen_x, screen_y, self.size)

    def check_hover(self, mouse_x, mouse_y, screen_x, screen_y):
        return screen_x - self.size <= mouse_x <= screen_x + self.size and screen_y - self.size <= mouse_y <= screen_y + self.size

class Cube:
    def __init__(self, width, height, length):
        self.width = width
        self.height = height
        self.length = length
        self.nodes = self.create_nodes()

    def create_nodes(self):
        nodes = []
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.length):
                    position = np.array([x - self.width / 2 + 0.5, y - self.height / 2 + 0.5, z - self.length / 2 + 0.5])
                    nodes.append(Node(position, color=(200, 200, 250)))  # Light blue color
        return nodes

    def dissect(self):
        group1 = [node for node in self.nodes if node.isOnDissectionPath]
        group2 = [node for node in self.nodes if not node.isOnDissectionPath]
        return group1, group2

    def draw(self, screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, mouse_x, mouse_y, dissected):
        sorted_nodes = sorted(self.nodes, key=lambda node: node.position.dot(rotation_matrices[2])[2], reverse=True)
        for node in sorted_nodes:
            screen_x, screen_y, size = node.draw(screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, dissected)
            if dissection_mode and node.check_hover(mouse_x, mouse_y, screen_x, screen_y):
                node.isOnDissectionPath = True

def draw_group_with_hull(screen, group, zoom, offset_x, offset_y, rotation_matrices):
    positions = np.array([node.position.dot(rotation_matrices[0]).dot(rotation_matrices[1]).dot(rotation_matrices[2])[:2] for node in group])
    positions[:, 0] = positions[:, 0] * zoom + offset_x
    positions[:, 1] = positions[:, 1] * zoom + offset_y

    if len(positions) > 3:
        hull = ConvexHull(positions)
        hull_points = positions[hull.vertices]
        pygame.draw.aalines(screen, (50, 150, 255), True, hull_points)  # Light blue outline

    for node in group:
        node.draw(screen, zoom, offset_x, offset_y, rotation_matrices, False, True)

# Cube dimensions
cube_width, cube_height, cube_length = 15, 15, 15
cube = Cube(cube_width, cube_height, cube_length)


# Rotation angles, zoom level, and flags
rotation_x, rotation_y, rotation_z = 0, 0, 0
zoom = 20
offset_x, offset_y = width // 2, height // 2
dragging = False
dissection_mode = False
dissected = False
new_objects = []

# Game loop
running = True
while running:
    mouse_x, mouse_y = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                dissection_mode = not dissection_mode
            elif event.key == pygame.K_RETURN and dissection_mode:
                group1, group2 = cube.dissect()
                new_objects = [group1, group2]
                dissected = True
                print(f"Group 1 has {len(group1)} nodes, Group 2 has {len(group2)} nodes.")
            elif event.key == pygame.K_w:
                rotation_x += 0.1  # Rotate around X-axis
            elif event.key == pygame.K_s:
                rotation_x -= 0.1
            elif event.key == pygame.K_a:
                rotation_y += 0.1  # Rotate around Y-axis
            elif event.key == pygame.K_d:
                rotation_y -= 0.1
            elif event.key == pygame.K_q:
                rotation_z += 0.1  # Rotate around Z-axis
            elif event.key == pygame.K_e:
                rotation_z -= 0.1
            elif event.key == pygame.K_UP:
                zoom += 2  # Zoom in
            elif event.key == pygame.K_DOWN:
                zoom = max(5, zoom - 2)  # Zoom out with a minimum limit
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging = False
        elif event.type == pygame.MOUSEMOTION and dragging:
            dx, dy = event.rel
            offset_x += dx
            offset_y += dy

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

    screen.fill((30, 30, 30))  # Dark background for contrast

    if dissected:
        draw_group_with_hull(screen, new_objects[0], zoom, offset_x - width // 4, offset_y, rotation_matrices)
        draw_group_with_hull(screen, new_objects[1], zoom, offset_x + width // 4, offset_y, rotation_matrices)
    else:
        cube.draw(screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, mouse_x, mouse_y, dissected)

    pygame.display.flip()

pygame.quit()
