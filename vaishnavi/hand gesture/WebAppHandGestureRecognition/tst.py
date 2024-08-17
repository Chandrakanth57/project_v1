import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os



def extract_hog_features(images):
    hog_features = []
    for image in images:
        features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(features)
    return np.array(hog_features)


def predict_image(image_path):
    new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hog_features1 = extract_hog_features([new_image])
    prediction = model.predict(hog_features1)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]


imp = r"C:\Users\Chandrakanth\Desktop\project\vaishnavi\hand gesture\dataset\test\13\919.jpg"
predicted_label = predict_image(imp)
print(f'Predicted Label: {predicted_label}')
