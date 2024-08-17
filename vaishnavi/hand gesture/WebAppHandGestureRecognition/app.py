

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from skimage import color
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['STATIC_FOLDER'] = 'static'

svm_classifier = joblib.load('svm_model.pkl')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_hog_features(images):
    hog_features = []
    for image in images:
        features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(features)
    return np.array(hog_features)



def classify_image(image):
    hog_features = extract_hog_features(image)
    prediction = svm_classifier.predict([hog_features])
    return prediction[0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        test_image = cv2.imread(filepath)
        if test_image is None:
            return "Error: Unable to load image."

        predicted_label = classify_image(test_image)

        # Save the original image and prediction
        static_image_path = os.path.join(app.config['STATIC_FOLDER'], filename)
        cv2.imwrite(static_image_path, test_image)

        return render_template('result.html', filename=filename, prediction=predicted_label)
    else:
        return redirect(request.url)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['STATIC_FOLDER']):
        os.makedirs(app.config['STATIC_FOLDER'])
    app.run(debug=True)
