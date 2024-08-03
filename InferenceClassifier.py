import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up Mediapipe hands with a static image mode and a minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map prediction labels to corresponding gestures
labels_dict = {0: 'Hi', 1: 'Ok', 2: 'Fit', 3: 'Bye', 4: 'Please'}

while True:
    data_aux = [] # List to hold the normalized coordinates of hand landmarks
    x_ = [] # List to hold x-coordinates of hand landmarks
    y_ = [] # List to hold y-coordinates of hand landmarks

    # Capture a frame from the webcam
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))  # Resize the frame to 800x600

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Extract and normalize hand landmark coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure data_aux has the correct length of 42
        if len(data_aux) == 42:
            # Calculate bounding box coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Make a prediction using the model
            prediction = model.predict([np.asarray(data_aux)])

            # Get the predicted gesture label
            predicted_character = labels_dict[int(prediction[0])]

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            
    # Display the frame with annotations
    cv2.imshow('frame', frame)
    
    # Check for the 'f' key to close the windows
    if cv2.waitKey(1) & 0xFF == ord('f'):
        break


# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
