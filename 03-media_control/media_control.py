import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import sys
import keras.models as models
from time import sleep
import pynput
import mediapipe as mp

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

LABELS = ['none', 'ok', 'stop', 'like', 'dislike']

# image size
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)
RESIZE_WIDTH =  480
RESIZE_HEIGHT = 270

cap = cv2.VideoCapture(video_id)

model = models.load_model("./gesture_recognition.keras")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Functionality for (just) hand position detection created by Github Copilot
def check_gesture(frame):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands using MediaPipe Hands
    results = hands.process(rgb_image)

    # resize image for faster processing
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Check if any hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box coordinates
            x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * RESIZE_WIDTH
            x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * RESIZE_WIDTH
            y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * RESIZE_HEIGHT
            y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * RESIZE_HEIGHT

            # Extract the ROI
            roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            if roi.size != 0:
                # Resize the ROI to the input size expected by the gesture recognition model
                roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                prediction = model.predict(np.expand_dims(roi, axis=0))
                gesture = LABELS[np.argmax(prediction)]
                print(gesture, np.max(prediction))

                return gesture
    return 'none'

def use_gesture(gesture):
    if gesture == 'ok':
        pynput.keyboard.Controller().press(pynput.keyboard.Key.media_play_pause)
    elif gesture == 'stop':
        pynput.keyboard.Controller().press(pynput.keyboard.Key.media_play_pause)
    elif gesture == 'like':
        print('would make louder now')
        #pynput.keyboard.Controller().press(pynput.keyboard.Key.media_volume_up)
    elif gesture == 'dislike':
        print('would make less louder now')
        #pynput.keyboard.Controller().press(pynput.keyboard.Key.media_volume_down)


while True:
    ret, frame = cap.read()
    gesture = check_gesture(frame)
    use_gesture(gesture)
    sleep(0.5)
