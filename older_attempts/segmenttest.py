import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Canvas to draw on
canvas = np.zeros((480, 640, 3), dtype="uint8")

# Background reference image (to be captured)
background = None

# Variables to keep track of previous positions for each finger for both hands
prev_positions = {
    'left': {
        'thumb': (None, None),
        'index': (None, None),
        'middle': (None, None),
        'ring': (None, None),
        'pinky': (None, None)
    },
    'right': {
        'thumb': (None, None),
        'index': (None, None),
        'middle': (None, None),
        'ring': (None, None),
        'pinky': (None, None)
    }
}

# Dictionary to map finger names to corresponding landmark indices
finger_tips = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20
}

# Buffer for temporary detection failure
detection_buffer = 5
no_detection_counter = {'left': 0, 'right': 0}

# Capture the background once before starting the main loop
ret, frame = cap.read()
if ret:
    background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(background, (21, 21), 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame so it acts as a mirror
    frame = cv2.flip(frame, 1)

    # Convert the frame color from BGR to RGB and grayscale for differencing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Perform image differencing
    frame_diff = cv2.absdiff(background, gray_frame)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the threshold image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to ignore small noise
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # You can adjust the area threshold
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]  # Region of interest
            
            # Use Mediapipe to process the region of interest if a hand is detected
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            result = hands.process(roi_rgb)
            
            if result.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    hand_label = 'right' if i == 0 else 'left'  # Label hands based on index
                    no_detection_counter[hand_label] = 0  # Reset the counter when hand is detected
                    
                    # Loop through each finger
                    for finger, tip_index in finger_tips.items():
                        # Get the coordinates of the current finger tip
                        tip = hand_landmarks.landmark[tip_index]
                        h, w, _ = roi.shape
                        x_tip, y_tip = int(tip.x * w), int(tip.y * h)
                        
                        # Adjust the coordinates to match the original frame
                        x_tip += x
                        y_tip += y
                        
                        # Get previous position of the finger
                        prev_x, prev_y = prev_positions[hand_label][finger]

                        # If previous position is None, initialize it
                        if prev_x is None and prev_y is None:
                            prev_positions[hand_label][finger] = (x_tip, y_tip)
                            continue

                        # Draw on the canvas by connecting previous and current points
                        cv2.line(canvas, (prev_x, prev_y), (x_tip, y_tip), (255, 0, 0), 5)

                        # Update the previous position of the finger
                        prev_positions[hand_label][finger] = (x_tip, y_tip)
                    
                    # Draw the landmarks on the hand
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combine the frame and canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the result
    cv2.imshow("Draw with Both Hands", combined)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
