import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create a hands object with MediaPipe
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory where data is saved
DATA_DIR = './data'

data = [] # List to store hand landmarks data
labels = [] # List to store corresponding labels

# Iterate through each class directory
for dir_ in os.listdir(DATA_DIR):
    # Iterate through each image in the class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = [] # List to store auxiliary data for each image
        x_ = [] # List to store x-coordinates of landmarks
        y_ = [] # List to store y-coordinates of landmarks

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path)) # Read the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image to RGB

        results = hands.process(img_rgb) # Process the image to detect hand landmarks
        if results.multi_hand_landmarks: # Check if hand landmarks are detected
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x) # Append x-coordinate to the list
                    y_.append(y) # Append y-coordinate to the list

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                     # Normalize coordinates and append to auxiliary data list
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux) # Append auxiliary data to main data list
            labels.append(dir_) # Append the corresponding label

# Save the data and labels to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
