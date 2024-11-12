import cv2
import numpy as np
from collections import deque
import mediapipe as mp

def create_hand_tracker():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7
    )
    return hands

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
        alpha = (radius - r) / radius  # Decreasing alpha for outer circles
        overlay = img.copy()
        img = cv2.addWeighted(overlay, alpha * 0.1, img, 1 - alpha * 0.1, 0)
        cv2.circle(overlay, center, r, color, -1)

    # Draw solid center
    cv2.circle(img, center, 5, color, -1)
    return img

def draw_skeleton(img, hand_landmarks):
    for idx in range(21):
        landmark = hand_landmarks.landmark[idx]
        h, w, _ = img.shape
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw landmark

    # Draw connections between landmarks
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
        cv2.line(img, start_point, end_point, (255, 0, 0), 2)  # Draw connections

def main():
    cap = cv2.VideoCapture(0)
    hands = create_hand_tracker()
    frame_count = 0  # Initialize frame count for image filenames

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

        # Capture 120 frames
        for _ in range(60):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = np.zeros_like(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[hand_idx].classification[0].label.lower()
                    prefix = 'l_' if handedness == 'left' else 'r_'

                    # Track and draw finger tips
                    for finger, tip_idx in finger_indices.items():
                        finger_name = f"{prefix}{finger}"
                        color = finger_colors_bgr[finger_name]

                        # Get fingertip coordinates
                        tip = hand_landmarks.landmark[tip_idx]
                        x = int(tip.x * frame.shape[1])
                        y = int(tip.y * frame.shape[0])

                        # Add point to corresponding finger's deque
                        finger_points[finger_name].appendleft((x, y))

                        # Draw glowing circle at fingertip
                        display = draw_glowing_circle(display, (x, y), color)

            # Apply fade effect to canvas
            canvas = cv2.multiply(canvas, 0.95)

            # Combine canvas with finger tips display
            result = cv2.addWeighted(display, 1, canvas, 0.7, 0)

            # Accumulate the result for overlapping
            overlapped_image = cv2.add(overlapped_image, result)

            cv2.imshow('Finger Tips Tracker', result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Draw the skeleton for the last captured frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_skeleton(overlapped_image, hand_landmarks)

        # Save the overlapped image
        filename = f'overlapped_image_{frame_count}.png'
        cv2.imwrite(filename, overlapped_image)
        print(f"Saved: {filename}")
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
