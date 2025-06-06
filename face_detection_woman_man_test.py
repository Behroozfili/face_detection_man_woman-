import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import load


def face_detector(img):
    try:
      detector = MTCNN()
      rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      faces = detector.detect_faces(rgb_img)
      out = faces[0]
      x, y, w, h = out["box"]
      return img[y:y+h, x:x+w],x,y,w,h
    except:
      pass
    
clf=load("gender_classifier.z")
# Path to the folder
folder_path = Path(r"E:\machinlerning\face_detection_man_woman\face_detection_test")

# Find all images with jpg and png extensions
image_paths = glob.glob(str(folder_path / '**' / '*.[jp][pn]g'), recursive=True)

# Process each image
for i, item in enumerate(image_paths):
    img = cv2.imread(item)
    face,x,y,w,h = face_detector(img)
    # If a face is detected, display it
    if face is None :continue
    face=cv2.resize(face,(32,32))
    face=face.flatten()
    face=face/255
    out=clf.predict(np.array([face]))[0]
    if out=="male":
       cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
       cv2.putText(img,"You are not allowed to enter",(x,y-10),cv2.FONT_HERSHEY_PLAIN,1.2,(0,0,255),2)
    if out=="female":
       cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
       cv2.putText(img,"You are allowed to enter",(x,y-20),cv2.FONT_HERSHEY_PLAIN,1.2,(0,255,0),2)
    
    cv2.imshow("image",img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
    
    

  
