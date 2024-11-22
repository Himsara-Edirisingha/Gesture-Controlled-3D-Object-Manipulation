import pygame
import numpy as np
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive Node Sphere with Enhanced Rendering")

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

class Sphere:
    def __init__(self, radius, resolution):
        self.radius = radius
        self.resolution = resolution
        self.nodes = self.create_nodes()

    def create_nodes(self):
        nodes = []
        phi, theta = np.mgrid[0.0:np.pi:self.resolution*1j, 0.0:2.0*np.pi:self.resolution*1j]
        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = self.radius * np.cos(phi)

        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                position = np.array([x[i, j], y[i, j], z[i, j]])
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

# Sphere dimensions
sphere_radius = 7
sphere_resolution = 20
sphere = Sphere(sphere_radius, sphere_resolution)

# Rotation angles, zoom level, and flags
rotation_x, rotation_y, rotation_z = 0, 0, 0
zoom = 40
offset_x, offset_y = width // 2, height // 2
dragging = False
dissection_mode = False
dissected = False
new_objects = []
fingertip_x = None
fingertip_y = None
sensitivity = 5

clock = pygame.time.Clock()
# loop
running = True
while running:
    mouse_x, mouse_y = pygame.mouse.get_pos()
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)  # Flip for natural interaction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            # Fingertip coordinates
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                fingertip_x = int(index_finger_tip.x * width)
                fingertip_y = int(index_finger_tip.y * height)

                # Draw fingertip position
                pygame.draw.circle(screen, (255, 0, 0), (fingertip_x, fingertip_y), sensitivity)

                # Map fingertip to sphere
                for node in sphere.nodes:
                    screen_x, screen_y, _ = node.draw(screen, zoom, offset_x, offset_y, rotation_matrices, False, dissected)
                    if abs(fingertip_x - screen_x) < sensitivity and abs(fingertip_y - screen_y) < sensitivity and dissection_mode:
                        node.isOnDissectionPath = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                dissection_mode = not dissection_mode
            elif event.key == pygame.K_RETURN and dissection_mode:
                group1, group2 = sphere.dissect()
                new_objects = [group1, group2]
                dissected = True
                print(f"Group 1 has {len(group1)} nodes, Group 2 has {len(group2)} nodes.")
            elif event.key == pygame.K_w:
                rotation_x += 0.1  # Rotate around x
            elif event.key == pygame.K_s:
                rotation_x -= 0.1
            elif event.key == pygame.K_a:
                rotation_y += 0.1  # Rotate around y
            elif event.key == pygame.K_d:
                rotation_y -= 0.1
            elif event.key == pygame.K_q:
                rotation_z += 0.1  # Rotate around z
            elif event.key == pygame.K_e:
                rotation_z -= 0.1
            elif event.key == pygame.K_UP:
                zoom += 2  # Zoom in
            elif event.key == pygame.K_DOWN:
                zoom = max(5, zoom - 2)  # Zoom out and minimum limit
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
    # bg
    screen.fill((200, 200, 200))

    if dissected:
        # Draw the dissected groups
        for i, group in enumerate(new_objects):
            offset = offset_x - width // 4 if i == 0 else offset_x + width // 4
            for node in group:
                node.draw(screen, zoom, offset, offset_y, rotation_matrices, False, True)
        fingertip_x = None
        fingertip_y = None
    else:
        sphere.draw(screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, mouse_x, mouse_y, dissected)

    if fingertip_x is not None and fingertip_y is not None:
        pygame.draw.circle(screen, (255, 0, 0), (fingertip_x, fingertip_y), 2)  # fingertip view
        fingertip_x = None
        fingertip_y = None

    pygame.display.flip()
    clock.tick(30)

cap.release()
hands.close()

pygame.quit()
