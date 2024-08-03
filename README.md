# Sign-Language-Detector-Project
Sign language detector with Python, OpenCV and Mediapipe!

Hello, I'm excited to share My Sign Language Detector Project! 🤟✨
In this project uses Python, OpenCV, and MediaPipe to detect and recognize sign language gestures. It processes webcam input to identify hand landmarks and translates gestures into text. 🚀🎉

🔧 Tools and Libraries Used:
- os: Provides functions for interacting with the operating system, such as file and directory management.
- pickle: Serializes and deserializes Python objects for saving and loading data.
- mediapipe: Offers machine learning solutions for computer vision tasks like hand and face detection.
- cv2: Part of OpenCV, used for image processing and computer vision tasks.
- matplotlib.pyplot: A plotting library for creating static, animated, and interactive visualizations in Python.
- numpy: Supports large, multi-dimensional arrays and matrices, along with mathematical functions to operate on them.
- scikit-learn: Provides simple and efficient tools for data mining and machine learning, including algorithms and preprocessing functions.

🛠 Project Workflow:
1. Data Collection (Collect Image Data) - 
In this part captures and saves images from a webcam for a dataset, organized into multiple classes. It first ensures a data directory exists, then opens the default camera. For each class, it creates a subdirectory and waits for the user to press 'Q' to start capturing images. The camera feed is resized and displayed with a prompt until 'Q' is pressed. Then, it captures a specified number of images, saving each one in the corresponding class directory. After capturing images for all classes, it releases the camera and closes all OpenCV windows. This process facilitates the collection of training data for machine learning purposes.

2. Data Processing (Create Data Set) -
In this Data Processing part processes images from a dataset to extract hand landmarks using MediaPipe, and saves the processed data for machine learning. It initializes MediaPipe's hand detection and drawing utilities, then iterates through each class directory and image within it. Each image is converted to RGB and processed to detect hand landmarks. If landmarks are found, their x and y coordinates are normalized and stored in a list. This landmark data is collected into a main data list, with corresponding labels. Finally, the data and labels are saved in a pickle file for future use in training or analysis.

3. Train Model (Train Classifier) - 
In this part trains a Random Forest classifier on hand landmark data to classify different classes. It starts by loading the preprocessed data from a pickle file, ensuring all data entries have a uniform shape by padding or truncating them to the same length. The labels are converted to a numpy array. The data is split into training and testing sets. A Random Forest classifier is then trained on the training set. The model's performance is evaluated on the test set, and the accuracy is printed. Finally, the trained model is saved to a pickle file for future use.

4. Test Model (Inference Classifier) - 
In this part uses a pre-trained Random Forest model to recognize hand gestures in real-time using a webcam. It initializes MediaPipe for hand landmark detection and captures frames from the webcam, resizing them to 800x600. Each frame is converted to RGB and processed to detect hand landmarks, which are drawn on the frame. The coordinates of the landmarks are normalized and used as input to the model, which predicts the gesture. The corresponding label is then displayed on the frame with a bounding box. The annotated frame is shown in a window, which closes when the 'f' key is pressed.
