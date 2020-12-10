# coding=utf-8
import os
import cv2
import numpy as np

# Keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras import regularizers
from keras import Model, Sequential
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import keras
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Models/vgg16_classWeights_clahe_98%_80%.h5'
vgg16 = VGG16(weights='imagenet')
x  = vgg16.get_layer('fc2').output
prediction = Dense(5, activation='softmax', name='predictions')(x)
model = Model(inputs=vgg16.input, outputs=prediction)

# Load your trained model
model.load_weights(MODEL_PATH)
graph = tf.get_default_graph()
sess = keras.backend.get_session()


print('Model loaded. Open loacalhost:5000')

def apply_clahe(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


def model_predict(img_path):
    img = cv2.imread(img_path)
    img = apply_clahe(img)
    img = preprocess_input(img)
    x = np.expand_dims(img, axis=0)
    #Forward pass for prediction
    preds = model.predict(x)
    return preds

def decode_prediction(preds):
    class_dict = { 0 : "Mild Diabetic Retinopathy",
                   1 : "Moderate Diabetic Retinopathy",
                   2 : "No Diabetic Retinopathy",
                   3 : "Proliferate Diabetic Retinopathy",
                   4 : "Severe Diabetic Retinopathy"}
    class_pred = preds.argmax(axis=-1)
    class_name = class_dict[class_pred[0]]
    return class_name


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file in main directory
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        global graph
        global sess
        with graph.as_default():
            set_session(sess)
            preds = model_predict(file_path)
            pred_class = decode_prediction(preds)
            return pred_class
    return None


if __name__ == '__main__':
    app.run(debug=True)
