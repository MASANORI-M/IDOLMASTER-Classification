from flask import Flask, redirect, request, jsonify, render_template
from tensorflow.keras import models
from PIL import Image
from keras.models import load_model
from flask_cors import CORS
from PIL import ImageFile
from keras.backend import tensorflow_backend as backend
import keras
import numpy as np
import sys, os, io
import glob
import tensorflow as tf

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
CORS(app)

imsize = (64, 64)
keras_param = "./cnn3.h5"

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize(imsize)
    img = np.asarray(img)
    img = img / 255.0

    return img

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            print('ファイルがありません')
        else:
            img = request.files["file"]
            graph = tf.get_default_graph()
            backend.clear_session()

            model = load_model(keras_param)
            img = load_image(img)
            pred = model.predict(np.array([img]))
            prelabel = np.argmax(pred, axis = 1)

            if prelabel == 0:
                print(">>> THE IDOLM@STER CINDERELLA GIRLS")
                name = "THE IDOLM@STER CINDERELLA GIRLS"
            elif prelabel == 1:
                print(">>> THE IDOLM@STER MILLION LIVE!")
                name = "THE IDOLM@STER MILLION LIVE!"

            return render_template('index.html', name = name)
    else:
        print("get request")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug = False, port = 5000)
