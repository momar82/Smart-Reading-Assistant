import cv2
import mediapipe as mp
import pytesseract
import time
from collections import deque
from statistics import mode, StatisticsError
import os


file_path = "scr/Dataset/vedio_3.mp4"

if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
else:
    print(f"The file '{file_path}' does not exist.")


# Initialize variables for text smoothing and handling
previous_text = ""
text_history = deque(maxlen=10)  # Adjust maxlen for smoothing sensitivity
holding_start = None  # Use None to indicate no holding is currently happening
word_printed = False  # Flag to check if the word has already been printed
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the video
video_cap = cv2.VideoCapture(file_path)

while True:
    ret, frame = video_cap.read()
    if not ret:
        break  # End of video reached

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            first_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            fingertip_x = int(first_fingertip.x * frame.shape[1])
            fingertip_y = int(first_fingertip.y * frame.shape[0])

            box_width, box_height = 200, 100
            top_left_x = max(0, fingertip_x - box_width // 2)
            top_left_y = max(0, fingertip_y - box_height - 20)
            bottom_right_x = min(frame.shape[1], top_left_x + box_width)
            bottom_right_y = top_left_y + box_height

            if top_left_x < frame.shape[1] and top_left_y < frame.shape[0]:
                text_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                text = pytesseract.image_to_string(text_region, config='--psm 6').lower().strip()
                text_history.append(text)

                try:
                    smoothed_text = mode(text_history)
                except StatisticsError:
                    smoothed_text = ""

                if smoothed_text == previous_text and smoothed_text != "":
                    if holding_start is None:
                        holding_start = time.time()
                        word_printed = False  # Reset the flag when a new holding starts

                    if time.time() - holding_start >= 1.0 and not word_printed:
                        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
                        print(f"Held word: {smoothed_text}")
                        word_printed = True  # Set the flag to prevent re-printing

                else:
                    holding_start = None  # Reset if not holding
                    word_printed = False  # Allow new word to be printed

                previous_text = smoothed_text

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
