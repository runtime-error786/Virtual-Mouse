import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

smoothening = 7
prev_x, prev_y = 0, 0

click_y_threshold = 15        

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

            index_x = int(landmarks.landmark[8].x * image.shape[1])
            index_y = int(landmarks.landmark[8].y * image.shape[0])
            middle_x = int(landmarks.landmark[12].x * image.shape[1])
            middle_y = int(landmarks.landmark[12].y * image.shape[0])

            index_x = prev_x + (index_x - prev_x) / smoothening
            index_y = prev_y + (index_y - prev_y) / smoothening

            prev_x, prev_y = index_x, index_y

            if index_x < middle_x and abs(index_y - middle_y) < click_y_threshold:
                pyautogui.click()

            pyautogui.moveTo(index_x, index_y)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
