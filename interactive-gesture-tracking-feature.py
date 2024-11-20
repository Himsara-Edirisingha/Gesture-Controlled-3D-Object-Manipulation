import threading
import time
import torch
from fastai.vision.all import *
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import pygame
from scipy.spatial import ConvexHull

####################################################################################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0) 

# Lighting configuration
light_positions = [
    np.array([0, -500, 0]),  # Light from front
    np.array([0, 500, 0]),   # Light from behind
    #np.array([-500, 500, 500])  # Light from top-left
]

class Node:
    def __init__(self, position, size=8, color=(128, 128, 128)):
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

    # if len(positions) > 3:
    #     hull = ConvexHull(positions)
    #     hull_points = positions[hull.vertices]
    #     pygame.draw.aalines(screen, (50, 150, 255), True, hull_points)  # Light blue outline

    for node in group:
        node.draw(screen, zoom, offset_x, offset_y, rotation_matrices, False, True)

####################################################################################################################
prediction = None

prediction_lock = threading.Lock()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learn = load_learner('model\\resnet-50_2024-10-30_08.06.14.pkl', cpu=device == torch.device('cpu'))

def create_hand_tracker():
    mp_hands = mp.solutions.hands
    gesturehands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7
    )
    return gesturehands

def rgb_to_bgr(color_name):
    color_map = {
        'red': (0, 0, 255),
        'lime': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'deeppink': (147, 20, 255),
        'lightskyblue': (250, 206, 135),
        'darksalmon': (122, 150, 233),
        'lightgreen': (144, 238, 144),
        'gold': (0, 215, 255),
        'darkviolet': (211, 0, 148)
    }
    return color_map[color_name]

def draw_glowing_circle(img, center, color, radius=15):
    for r in range(radius, 1, -2):
        alpha = (radius - r) / radius
        overlay = img.copy()
        img = cv2.addWeighted(overlay, alpha * 0.1, img, 1 - alpha * 0.1, 0)
        cv2.circle(overlay, center, r, color, -1)

    cv2.circle(img, center, 5, color, -1)
    return img

def draw_skeleton(img, hand_landmarks):
    for idx in range(21):
        landmark = hand_landmarks.landmark[idx]
        h, w, _ = img.shape
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw landmark

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    for connection in connections:
        start = hand_landmarks.landmark[connection[0]]
        end = hand_landmarks.landmark[connection[1]]
        h, w, _ = img.shape
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(img, start_point, end_point, (255, 0, 0), 2) 

def display_value():
    global prediction
    # Initialize Pygame
    pygame.init()
    # Screen dimensions
    width, height = 1000, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Interactive 3D cube Manipulation")

    # Cube dimensions
    cube_width, cube_height, cube_length = 10, 12, 8
    cube = Cube(cube_width, cube_height, cube_length)

    PREDICTION_EVENT = pygame.USEREVENT + 1 
    # Rotation angles, zoom level, and flags
    rotation_x, rotation_y, rotation_z = 0, 0, 0
    zoom = 10
    offset_x, offset_y = width // 2, height // 2
    dragging = False
    dissection_mode = False
    dissected = False
    new_objects = []
    fingertip_x = None
    fingertip_y = None
    sensitivity = 5

    running = True
    while running:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)  # Flip for avoid lateral inversion
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                # Extract fingertip coordinates
                for hand_landmarks in result.multi_hand_landmarks:
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    fingertip_x = int(index_finger_tip.x * width)
                    fingertip_y = int(index_finger_tip.y * height)

                    # Draw fingertip position on Pygame screen for reference
                    pygame.draw.circle(screen, (255, 0, 0), (fingertip_x, fingertip_y), sensitivity)

                    # Map fingertip to 3D cube
                    for node in cube.nodes:
                        screen_x, screen_y, _ = node.draw(screen, zoom, offset_x, offset_y, rotation_matrices, False, dissected)
                        if abs(fingertip_x - screen_x) < sensitivity and abs(fingertip_y - screen_y) < sensitivity and dissection_mode:
                            node.isOnDissectionPath = True

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

        with prediction_lock:
            #print(f"Value updated to on desplay thread: {prediction}")
            #Pointing
            if prediction == 'Blank':
                 zoom = min(10, zoom + 2)  
                 prediction = None          
            elif prediction == 'Resting':
                prediction = None
            elif prediction == 'Rotating':
                for x in range(6):
                    rotation_x += 0.001 
            elif prediction == 'Pointing-With-Hand-Raised':
                for x in range(6):
                    rotation_y += 0.001 
            elif prediction == 'Catching-Hands-Up':
                for x in range(6):
                    rotation_z -= 0.001 
            elif prediction == 'Pointing':
                    #zoom = max(5, zoom - 2) 
                    prediction = None
            elif prediction == 'C':
                    #zoom = max(5, zoom - 2)  
                    prediction = None
                    
            pygame.event.post(pygame.event.Event(PREDICTION_EVENT))
            #Catching-Hands-Up  
            #Pointing-With-Hand-Raised
            #Rotating
            #Zoom
            #Shaking-Raised-Fist
            #Catching

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

            screen.fill((200, 200, 200))  # Dark background for contrast

            if dissected:
                draw_group_with_hull(screen, new_objects[0], zoom, offset_x - width // 4, offset_y, rotation_matrices)
                draw_group_with_hull(screen, new_objects[1], zoom, offset_x + width // 4, offset_y, rotation_matrices)
                fingertip_x = None
                fingertip_y = None
            else:
                cube.draw(screen, zoom, offset_x, offset_y, rotation_matrices, dissection_mode, mouse_x, mouse_y, dissected)

            if fingertip_x is not None and fingertip_y is not None:
                pygame.draw.circle(screen, (255, 0, 0), (fingertip_x, fingertip_y), 2)  # fingertip view
                fingertip_x = None
                fingertip_y = None

            pygame.display.flip()

    cap.release()
    hands.close()
    pygame.quit()


def predict():
    global prediction
    cap = cv2.VideoCapture(0)
    gesturehands = create_hand_tracker()
    frame_count = 0 

    while True:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        overlapped_image = np.zeros((480, 640, 3), dtype=np.uint8)

        finger_points = {
            'l_thumb': deque(maxlen=64),
            'l_index': deque(maxlen=64),
            'l_middle': deque(maxlen=64),
            'l_ring': deque(maxlen=64),
            'l_pinky': deque(maxlen=64),
            'r_thumb': deque(maxlen=64),
            'r_index': deque(maxlen=64),
            'r_middle': deque(maxlen=64),
            'r_ring': deque(maxlen=64),
            'r_pinky': deque(maxlen=64)
        }

        fingers_colors = {
            'l_thumb': 'red',
            'l_index': 'lime',
            'l_middle': 'blue',
            'l_ring': 'yellow',
            'l_pinky': 'deeppink',
            'r_thumb': 'lightskyblue',
            'r_index': 'darksalmon',
            'r_middle': 'lightgreen',
            'r_ring': 'gold',
            'r_pinky': 'darkviolet'
        }

        finger_colors_bgr = {k: rgb_to_bgr(v) for k, v in fingers_colors.items()}

        finger_indices = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }

        for _ in range(90):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = np.zeros_like(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = gesturehands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[hand_idx].classification[0].label.lower()
                    prefix = 'l_' if handedness == 'left' else 'r_'

                    for finger, tip_idx in finger_indices.items():
                        finger_name = f"{prefix}{finger}"
                        color = finger_colors_bgr[finger_name]
                        tip = hand_landmarks.landmark[tip_idx]
                        x = int(tip.x * frame.shape[1])
                        y = int(tip.y * frame.shape[0])
                        finger_points[finger_name].appendleft((x, y))
                        display = draw_glowing_circle(display, (x, y), color)

            canvas = cv2.multiply(canvas, 0.95)

            result = cv2.addWeighted(display, 1, canvas, 0.7, 0)

            overlapped_image = cv2.add(overlapped_image, result)
            with prediction_lock:
                if prediction is not None:
                    cv2.putText(result, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Finger Tips Tracker', result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_skeleton(overlapped_image, hand_landmarks)
        frame_count += 1
        img = overlapped_image
        pred, pred_idx, probs = learn.predict(img)
        print(f"Prediction: {pred}, Probability: {probs[pred_idx]:.4f}")
        with prediction_lock:
            prediction = pred
            print(f"Value updated to: {prediction}")

prediction_thread = threading.Thread(target=predict)
shape_manipulation_thread = threading.Thread(target=display_value)

prediction_thread.start()
shape_manipulation_thread.start()

prediction_thread.join()
shape_manipulation_thread.join()
