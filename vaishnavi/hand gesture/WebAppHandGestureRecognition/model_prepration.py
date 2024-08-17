import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

def load_images_from_folder(folder):

    images = []
    labels = []
    for label in os.listdir(folder):
        print(label)
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img = cv2.imread(os.path.join(label_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

def extract_hog_features(images):
    hog_features = []
    for image in images:
        features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(features)
    return np.array(hog_features)

path_location = r"C:\Users\Chandrakanth\Desktop\project\vaishnavi\hand gesture\dataset\train\train"
images, labels = load_images_from_folder(path_location)

hog_features = extract_hog_features(images)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(hog_features, labels_encoded, test_size=0.2, random_state=42)

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

def predict_image(image_path):
    new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hog_features1 = extract_hog_features([new_image])
    prediction = model.predict(hog_features1)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

imp = r"C:\Users\Chandrakanth\Desktop\project\vaishnavi\hand gesture\dataset\test\3\901.jpg"
predicted_label = predict_image(imp)
print(f'Predicted Label: {predicted_label}')


import joblib

# Save the model to a file
joblib_file = r"C:\Users\Chandrakanth\Desktop\project\vaishnavi\hand gesture\WebAppHandGestureRecognition\svm_model.pkl"
joblib.dump(model, joblib_file)