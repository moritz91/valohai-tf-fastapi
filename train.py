#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022  <@HBL1260W>
#
# Distributed under terms of the MIT license.

import numpy as np
import tensorflow as tf
import valohai

valohai.prepare(
    step='train-model',
    image='tensorflow/tensorflow:2.6.0',
    default_inputs={
        'dataset': 'https://valohaidemo.blob.core.windows.net/mnist/mnist.npz'
    },
    default_parameters={
        'learning_rate': 0.001,
        'epoch': 10,
    },
)

def log_metadata(epoch, logs):
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])

input_path = valohai.inputs('dataset').path()

with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

output_path = valohai.outputs().path('model.h5')

model.save(output_path)
