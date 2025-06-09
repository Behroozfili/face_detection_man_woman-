# Gender Classification from Face Images

This project implements a machine learning pipeline to classify gender (male/female) from face images. It involves face detection using MTCNN, feature extraction, training a Stochastic Gradient Descent (SGD) classifier, and then using the trained model to detect faces in new images, predict gender, and display an access control message.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone Repository (Optional)](#1-clone-repository-optional)
  - [2. Create Virtual Environment (Recommended)](#2-create-virtual-environment-recommended)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Prepare Dataset](#4-prepare-dataset)
- [Usage](#usage)
  - [Part 1: Training the Gender Classifier](#part-1-training-the-gender-classifier)
    - [1. Configure Paths (Training Script)](#1-configure-paths-training-script)
    - [2. Run the Training Script](#2-run-the-training-script)
  - [Part 2: Real-time Gender Detection and Access Control](#part-2-real-time-gender-detection-and-access-control)
    - [1. Prepare Test Images](#1-prepare-test-images)
    - [2. Configure Paths (Prediction Script)](#2-configure-paths-prediction-script)
    - [3. Run the Prediction Script](#3-run-the-prediction-script)
- [Workflow](#workflow)
  - [Training Phase](#training-phase)
  - [Prediction Phase](#prediction-phase)
- [File Descriptions](#file-descriptions)
- [Customization](#customization)
- [License](#license)

## Project Overview

The project consists of two main parts:
1.  **Training a Gender Classifier:** A script (`train_gender_classifier.py`) that:
    -   Loads images from a dataset structured by gender (e.g., folders named "male" and "female").
    -   Detects faces using MTCNN.
    -   Resizes, flattens, and normalizes the detected face images to create feature vectors.
    -   Trains an SGDClassifier on these features.
    -   Evaluates the classifier and saves the trained model.
2.  **Gender Detection and Access Simulation:** A script (`detect_gender_access.py`) that:
    -   Loads the pre-trained gender classifier.
    -   Processes images from a test folder.
    -   Detects faces using MTCNN.
    -   Predicts the gender of the detected face.
    -   Draws a bounding box around the face and displays a message ("You are allowed to enter" for female, "You are not allowed to enter" for male) on the image.

## Features

-   **Face Detection:** Utilizes the MTCNN (Multi-task Cascaded Convolutional Networks) library for robust face detection.
-   **Image Preprocessing:**
    -   Resizes detected faces to a consistent size (32x32 pixels).
    -   Flattens images into 1D feature vectors.
    -   Normalizes pixel values to the range [0, 1].
-   **Classification Model:** Employs `SGDClassifier` from Scikit-learn for gender classification.
-   **Model Persistence:** Saves and loads the trained classifier using `joblib`.
-   **Real-time Feedback (Simulation):** Annotates images with bounding boxes and access control messages based on predicted gender.
-   **Dataset Handling:** Recursively loads images from specified folders and uses parent folder names as labels.

## Dataset

**For Training (`train_gender_classifier.py`):**
You need a dataset of face images organized into subfolders by gender. The script expects a parent folder (e.g., "Gender") containing subfolders like "male" and "female", with respective images inside them.

Example structure:
Use code with caution.
Markdown
<project-root>/
├── Gender/
│ ├── male/
│ │ ├── male_face1.jpg
│ │ ├── male_face2.png
│ │ └── ...
│ └── female/
│ ├── female_face1.jpg
│ ├── female_face2.jpg
│ └── ...
**For Testing/Prediction (`detect_gender_access.py`):**
A folder containing images (can be mixed gender, single or multiple faces per image) on which you want to perform gender detection.

Example structure:
Use code with caution.
<project-root>/
├── face_detection_test/
│ ├── test_image1.jpg
│ ├── group_photo.png
│ └── ...
## Prerequisites

-   Python (3.7+ recommended)
-   pip (Python package installer)
-   Git (optional, if you clone a repository)

## Setup

### 1. Clone Repository (Optional)

If this code is part of a Git repository:
```bash
git clone <your-repository-url>
cd <your-repository-name>
Use code with caution.
Otherwise, save the Python scripts (e.g., train_gender_classifier.py and detect_gender_access.py) to your project directory.
2. Create Virtual Environment (Recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Use code with caution.
Bash
3. Install Dependencies
Create a requirements.txt file with the following content:
# requirements.txt
tensorflow # MTCNN has a dependency on TensorFlow
opencv-python
mtcnn
numpy
scikit-learn
joblib
Use code with caution.
Txt
Then, install the dependencies:
pip install -r requirements.txt
Use code with caution.
Bash
(Note: mtcnn library might install TensorFlow if not already present. Ensure TensorFlow is compatible with your system, especially if you have a GPU.)
4. Prepare Dataset
Training Data: Create the "Gender" folder (or your chosen name) with "male" and "female" subfolders as described in the Dataset section and populate them with images.
Test Data: Create a folder (e.g., "face_detection_test") and add images you want to test the prediction script on.
Usage
Part 1: Training the Gender Classifier
This part uses the script train_gender_classifier.py (or your equivalent file name).
1. Configure Paths (Training Script)
Open train_gender_classifier.py and ensure the folder_path variable points to your main training dataset directory:
# In train_gender_classifier.py
# folder_path = Path(r"E:\machinlerning\face_detection_man_woman\Gender")
folder_path = Path("Gender") # Example using relative path
Use code with caution.
Python
The trained model will be saved as gender_classifier.z in the same directory as the script.
2. Run the Training Script
Execute the training script from your terminal:
python train_gender_classifier.py
Use code with caution.
Bash
The script will process images, detect faces, and extract features. Progress will be printed.
After training, the accuracy of the classifier on the test set will be displayed.
The trained model gender_classifier.z will be saved.
Part 2: Real-time Gender Detection and Access Control
This part uses the script detect_gender_access.py (or your equivalent file name).
1. Prepare Test Images
Ensure you have a folder with images for testing (e.g., "face_detection_test").
2. Configure Paths (Prediction Script)
Open detect_gender_access.py:
Ensure the loaded model path is correct (it defaults to gender_classifier.z in the current directory).
Update the folder_path variable to point to your folder of test images:
# In detect_gender_access.py
# clf = load("gender_classifier.z") # Assumes model is in the same directory

# folder_path = Path(r"E:\machinlerning\face_detection_man_woman\face_detection_test")
folder_path = Path("face_detection_test") # Example using relative path
Use code with caution.
Python
3. Run the Prediction Script
Execute the prediction script from your terminal:
python detect_gender_access.py
Use code with caution.
Bash
The script will iterate through images in the specified test folder.
For each image:
It will attempt to detect a face.
If a face is detected, it predicts the gender.
An OpenCV window will pop up showing the original image with a bounding box around the detected face and an access message:
Male: Red box, "You are not allowed to enter".
Female: Green box, "You are allowed to enter".
Press any key to close the current image window and proceed to the next one.
After all images are processed, all OpenCV windows will be closed.
Workflow
Training Phase (train_gender_classifier.py)
Load Images: Recursively scan the dataset folder for images.
Face Detection: For each image, use MTCNN to detect the face region.
Preprocessing:
Crop the detected face.
Resize the face to 32x32 pixels.
Flatten the 2D face image into a 1D array.
Normalize pixel values (divide by 255.0).
Label Extraction: Extract gender labels from the parent directory names (e.g., "male", "female").
Data Splitting: Split the processed data (features and labels) into training and testing sets.
Model Training: Initialize and train an SGDClassifier on the training data.
Evaluation: Predict labels for the test set and calculate accuracy.
Save Model: Save the trained classifier to a file (gender_classifier.z) using joblib.
Prediction Phase (detect_gender_access.py)
Load Model: Load the pre-trained gender_classifier.z.
Load Test Images: Recursively scan the specified test image folder.
For each test image:
a. Face Detection: Use MTCNN to detect a face and get its bounding box coordinates.
b. Preprocessing: If a face is detected:
- Crop the face.
- Resize to 32x32.
- Flatten and normalize.
c. Prediction: Use the loaded classifier to predict the gender from the processed face features.
d. Annotation & Display:
- Draw a rectangle around the original face in the input image.
- Add text indicating the access status based on the predicted gender.
- Display the annotated image using OpenCV. Wait for a key press.
Cleanup: Close all OpenCV windows.
File Descriptions
train_gender_classifier.py (or similar): Script for training the gender classification model.
detect_gender_access.py (or similar): Script for using the trained model to detect gender in images and simulate access control.
requirements.txt: Lists all Python package dependencies.
gender_classifier.z: (Generated by training script) The saved, trained Scikit-learn SGDClassifier model.
Gender/ (User-provided): Directory containing training images, organized by gender subfolders.
face_detection_test/ (User-provided): Directory containing images for testing the prediction script.
Customization
Face Detector: While MTCNN is used, you could experiment with other face detectors (e.g., Haar cascades from OpenCV, dlib).
Image Size: The face images are resized to 32x32. You can change this in both scripts, but ensure consistency. Larger sizes might capture more detail but increase computation.
Classifier: SGDClassifier is used. You can try other Scikit-learn classifiers (e.g., SVC, RandomForestClassifier) or even deep learning models for gender classification.
Dataset Path: Crucially, update the folder_path variables in both scripts to match your local setup. Using relative paths (as suggested in the examples) is good practice.
Access Logic: The access messages in detect_gender_access.py are illustrative. You can modify the logic and messages as needed.
Error Handling: The face_detector function has basic try-except blocks. You might want to add more robust error handling or logging.

Important Notes for Users:
Ensure the gender_classifier.z model file generated by the training script is available in the location expected by the prediction script (by default, the same directory).
The MTCNN model weights will be downloaded automatically the first time it's used. This requires an internet connection.
Paths are hardcoded in the provided snippets (e.g., E:\...). The README encourages using relative paths and provides examples. Users must adjust these paths.
