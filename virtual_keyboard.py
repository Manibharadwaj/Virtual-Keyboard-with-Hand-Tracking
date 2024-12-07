import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller

# Initialize MediaPipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize keyboard controller
keyboard = Controller()

# Keyboard Layout and Dimensions
keyboard_layout = [
    "1234567890",
    "QWERTYUIOP",
    "ASDFGHJKL",
    "ZXCVBNM"
]
KEY_WIDTH = 40
KEY_HEIGHT = 40
SPACING = 8
START_X = 50
START_Y = 300

# Function to draw the keyboard
def draw_keyboard(frame):
    y = START_Y
    for row in keyboard_layout:
        x = START_X
        for key in row:
            cv2.rectangle(frame, (x, y), (x + KEY_WIDTH, y + KEY_HEIGHT), (200, 200, 200), -1)
            cv2.putText(frame, key, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            x += KEY_WIDTH + SPACING
        y += KEY_HEIGHT + SPACING

    # Add a "Back" button
    back_x = START_X
    back_y = START_Y - 50
    back_width = 80
    back_height = 40
    cv2.rectangle(frame, (back_x, back_y), (back_x + back_width, back_y + back_height), (255, 100, 100), -1)
    cv2.putText(frame, "Back", (back_x + 10, back_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return back_x, back_y, back_width, back_height

# Function to detect key pressed
def get_key_from_position(x, y):
    row_index = (y - START_Y) // (KEY_HEIGHT + SPACING)
    col_index = (x - START_X) // (KEY_WIDTH + SPACING)
    if 0 <= row_index < len(keyboard_layout) and 0 <= col_index < len(keyboard_layout[row_index]):
        return keyboard_layout[row_index][col_index]
    return None

# Main Function
def main():
    cap = cv2.VideoCapture(0)
    search_text = ""
    last_pressed_key = None
    key_pressed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip for mirror view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make the camera feed occupy the top half of the frame
        frame_height, frame_width, _ = frame.shape
        top_half = frame_height // 2
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Draw search bar with semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (frame_width - 50, 120), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, search_text, (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Draw the keyboard and back button
        back_x, back_y, back_width, back_height = draw_keyboard(frame)

        # Detect hands
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get fingertip position (index finger tip = landmark 8)
                x_tip = int(hand_landmarks.landmark[8].x * frame.shape[1])
                y_tip = int(hand_landmarks.landmark[8].y * frame.shape[0])

                # Highlight the fingertip
                cv2.circle(frame, (x_tip, y_tip), 10, (0, 255, 0), -1)

                # Check if fingertip is over the "Back" button
                if back_x < x_tip < back_x + back_width and back_y < y_tip < back_y + back_height:
                    cv2.rectangle(frame, (back_x, back_y), (back_x + back_width, back_y + back_height), (100, 255, 100), -1)
                    cv2.putText(frame, "Back", (back_x + 10, back_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if not key_pressed:
                        search_text = search_text[:-1]  # Simulate backspace
                        key_pressed = True

                # Check if fingertip is over a key
                key = get_key_from_position(x_tip, y_tip)
                if key and not key_pressed:
                    # Simulate a "tap" when fingertip is over a key
                    search_text += key
                    last_pressed_key = key
                    key_pressed = True

                # Reset key press when finger moves away
                if last_pressed_key and key != last_pressed_key:
                    key_pressed = False

        # Show the frame
        cv2.imshow("Virtual Keyboard", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
