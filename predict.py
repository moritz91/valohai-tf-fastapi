#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022  <@HBL1260W>
#
# Distributed under terms of the MIT license.

from fastapi import FastAPI, File, UploadFile

import tensorflow as tf

import numpy
from PIL import Image
from io import BytesIO

app = FastAPI()

model_path = 'model.h5'
loaded_model = None

@app.post("{full_path:path}")
async def predict(image: UploadFile = File(...)):
    img = Image.open(BytesIO(await image.read()))

    # Resize image and convert to grayscale
    img = img.resize((28, 28)).convert('L')
    img_array = numpy.array(img)

    image_data = numpy.reshape(img_array, (1, 28, 28))

    global loaded_model
    # Check if model is already loaded
    if not loaded_model:
        loaded_model = tf.keras.models.load_model(model_path)

    # Predict with the model
    prediction = loaded_model.predict_classes(image_data)

    return f'Predicted_Digit: {prediction[0]}'
