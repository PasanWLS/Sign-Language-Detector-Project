import os
import cv2

# Directory where data will be saved
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Create the data directory if it doesn't exist

number_of_classes = 5 # Number of different classes to collect data for
dataset_size = 100 # Number of images per class to collect


cap = cv2.VideoCapture(0) # Open the default camera
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j))) # Create a directory for each class if it doesn't exist

    print('Collecting data for class {}'.format(j))

    done = False
    # Wait for user to get ready and press 'Q' to start collecting images
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 600))  # Resize the frame to 600x600
        cv2.putText(frame, 'Ready? Press "Q" to Capture!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1 # Increment the counter

cap.release() # Release the camera
cv2.destroyAllWindows() # Close all OpenCV windows
