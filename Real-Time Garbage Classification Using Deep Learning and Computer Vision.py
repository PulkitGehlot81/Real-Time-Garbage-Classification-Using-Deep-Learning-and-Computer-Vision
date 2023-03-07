import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = keras.models.load_model('garbage_classification_model.h5')

# Set up paths and classes
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Set up camera capture
cap = cv2.VideoCapture(0)

# Loop over video frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict the class of garbage
    prediction = model.predict(img)
    class_idx = np.argmax(prediction[0])
    class_label = classes[class_idx]

    # Draw the class label on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, class_label, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Garbage Classification', frame)

    # Quit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
