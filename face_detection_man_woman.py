 # Import necessary libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
from pathlib import Path
import glob  # Used for file path matching
from sklearn.model_selection import train_test_split  # Split data into train/test sets
import cv2  # OpenCV for image processing
from mtcnn.mtcnn import MTCNN  # MTCNN for face detection
import numpy as np  # Numpy for array manipulation
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent Classifier
from sklearn.metrics import accuracy_score  # To calculate accuracy of the model
from joblib import dump  # To save the trained model

# Initialize lists for storing image data and corresponding labels
data = []
lables = []

# Function to detect faces in an image using MTCNN
def face_detector(img):
    try:
        detector = MTCNN()  # Initialize the MTCNN face detector
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
        faces = detector.detect_faces(rgb_img)  # Detect faces in the image
        out = faces[0]  # Get the first face detected
        x, y, w, h = out["box"]  # Get coordinates of the face
        return img[y:y+h, x:x+w]  # Return the cropped face region
    except:
        pass  # In case of any error, skip the image

# Path to the folder containing the dataset
folder_path = Path(r"E:\machinlerning\face_detection_man_woman\Gender")

# Find all image files (jpg and png) in the folder and subfolders
image_paths = glob.glob(str(folder_path / '**' / '*.[jp][pn]g'), recursive=True)

# Process each image
for i, item in enumerate(image_paths):
    img = cv2.imread(item)  # Read the image
    face = face_detector(img)  # Detect face in the image
    # If no face is detected, skip this image
    if face is None:
        continue
    face = cv2.resize(face, (32, 32))  # Resize the face to 32x32 pixels
    face = face.flatten()  # Flatten the 2D image into a 1D array
    face = face / 255.0  # Normalize the pixel values between 0 and 1
    data.append(face)  # Add the face data to the list
    lable = item.split("\\")[-2]  # Extract the label (gender) from the file path
    lables.append(lable)  # Append the label to the labels list
    # Display progress every 100 images
    if i % 100 == 0:
        print(f"[info] : {i}/{len(image_paths)} processed")

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(lables)

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(data, lables, test_size=0.2, random_state=42)

# Initialize the SGDClassifier (Stochastic Gradient Descent Classifier)
clf = SGDClassifier()

# Train the classifier on the training data
clf.fit(x_train, y_train)

# Predict the labels for the test data
y_predict = clf.predict(x_test)

# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_predict)
print("Accuracy : ", acc * 100)

# Save the trained model to a file
dump(clf, "gender_classifier.z")
