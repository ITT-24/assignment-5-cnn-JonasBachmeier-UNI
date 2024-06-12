import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import sys
import keras.models as models
from time import sleep

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# include only those gestures
CONDITIONS = ['ok', 'stop', 'like', 'dislike']
LABELS = ['none', 'ok', 'stop', 'like', 'dislike']

# image size
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)

# number of color channels we want to use
# set to 1 to convert to grayscale
# set to 3 to use color images
COLOR_CHANNELS = 3

cap = cv2.VideoCapture(video_id)

model = models.load_model("./gesture_recognition.keras")

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def check_gesture(frame):
    # Preprocess the image
    predictions = []
    # Go through the image with a sliding window
    for (x, y, window) in sliding_window(frame, stepSize=int(IMG_SIZE/2), windowSize=SIZE):
        if window.shape != (IMG_SIZE, IMG_SIZE, COLOR_CHANNELS):
            continue
        # Classify each region
        prediction = model.predict(np.expand_dims(window, axis=0))
        gesture = LABELS[np.argmax(prediction)]
        print(gesture, np.max(prediction))
        predictions.append(gesture)
    # Get the most second most common prediction
    print(predictions)
    return predictions

while True:
    ret, frame = cap.read()
    gesture = check_gesture(frame)
    print(CONDITIONS[gesture])
    sleep(2)
