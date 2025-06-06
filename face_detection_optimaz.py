import cv2
import glob
import numpy as np
from mtcnn import MTCNN
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Dataset and labels
data = []
labels = []

# Face detection function
def face_detector(img):
    detector = MTCNN()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
    faces = detector.detect_faces(rgb_img)  # Detect faces
    
    if len(faces) > 0:
        # Extract the first detected face
        x, y, w, h = faces[0]["box"]
        return img[y:y+h, x:x+w]  # Crop the face
    return None  # Return None if no face is found

# Function to process each image
def process_image(item):
    img = cv2.imread(item)  # Read the image
    face = face_detector(img)  # Detect the face
    if face is not None:
        face = cv2.resize(face, (32, 32))  # Resize the image to 32x32 pixels
        face = face.flatten()              # Flatten the image into a 1D array
        face = face / 255                  # Normalize the pixel values
        label = item.split("\\")[-2]       # Extract label from file path
        return face, label
    return None, None

# Path to the image folder
folder_path = Path(r"E:\machinlerning\face_detection_man_woman\Gender")

# Find all images with jpg and png extensions
image_paths = glob.glob(str(folder_path / '**' / '*.[jp][pn]g'), recursive=True)

# Process images in parallel using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    results = executor.map(process_image, image_paths)

# Collect data and labels
for i,(face, label) in enumerate(results):
    if face is not None:
        data.append(face)
        labels.append(label)
        if i%100==0:
          print(f"[info] : {i}/{len(image_paths)} proccesed")

        

# Convert lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the classifier
clf = SGDClassifier()
clf.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = clf.predict(x_test)
acc = accuracy_score(y_test, y_predict)

# Display the accuracy
print(f"Accuracy: {acc * 100:.2f}%")

# Save the trained model
dump(clf, "gender_classifier.joblib")
