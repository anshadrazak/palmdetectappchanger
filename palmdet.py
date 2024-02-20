import cv2
import mediapipe as mp
import pyautogui

state = 1

# Set up MediaPipe solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # Detect only one hand
    min_detection_confidence=0.5,  # Adjust as needed
    min_tracking_confidence=0.5  # Adjust as needed
)
mp_drawing = mp.solutions.drawing_utils

# Define initial state as closed palm
palm_state = "closed"

# Analyze gesture function
def analyze_gesture(hand_landmarks):
    # Implement your gesture logic here (distance, angles, etc.)
    thumb_tip = hand_landmarks.landmark[4]
    index_finger_tip = hand_landmarks.landmark[8]
    distance = calculate_distance(thumb_tip, index_finger_tip)
    return distance > 0.15  # Adjust threshold for open palm

# Distance calculation function
def calculate_distance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# Access webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert image to RGB format for MediaPipe
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image for hand landmark detection
    results = hands.process(image)

    # Check for detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Analyze gesture and update state
            is_open_palm = analyze_gesture(hand_landmarks)
            if is_open_palm and palm_state == "closed":
                print("Open palm detected!")
                if state == 1 :
                    pyautogui.hotkey('space')  # Simulate Alt + Tab key press
                    state = 2
                else :
                    pyautogui.hotkey('space')  # Simulate Alt + Tab key press
                    state = 1
                palm_state = "open"  # Update state to open
            elif not is_open_palm and palm_state == "open":
                palm_state = "closed"  # Update state to closed

            # Perform actions based on updated state (if needed)

    # Display the image
    cv2.imshow('Hand Gesture Recognition', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
