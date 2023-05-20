from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pandas as pd
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def loadImages(path, urls, target):
  images = []
  labels = []
  for i in range(len(urls)):
    img_path = os.path.join(path, urls[i])
    img = cv2.imread(img_path)
    img = img / 255.0
    img = cv2.resize(img, (100, 100))
    images.append(img)
    labels.append(target)
  images = np.asarray(images)
  return images, labels

covid_path = "COVID-19_Radiography_Dataset/COVID/images"
covidUrls = os.listdir(covid_path)
covidImages, covidTargets = loadImages(covid_path, covidUrls,[0,1,0,0])

normal_path = "COVID-19_Radiography_Dataset/Normal/images"
normalUrls = os.listdir(normal_path)
normalImages, normalTargets = loadImages(normal_path, normalUrls, [1,0,0,0])

opacity_path = "COVID-19_Radiography_Dataset/Lung_Opacity/images"
opacityUrls = os.listdir(opacity_path)
opacityImages, opacityTargets = loadImages(opacity_path, opacityUrls,[0,0,1,0])

pneumonia_path = "COVID-19_Radiography_Dataset/Viral Pneumonia/images"
pneumoniaUrls = os.listdir(pneumonia_path)
pneumoniaImages, pneumoniaTargets = loadImages(pneumonia_path, pneumoniaUrls,[0,0,0,1])

covidImages = np.asarray(covidImages)
normalImages = np.asarray(normalImages)
opacityImages = np.asarray(opacityImages)
pneumoniaImages = np.asarray(pneumoniaImages)

data = np.concatenate([covidImages, normalImages, opacityImages, pneumoniaImages])
targets = np.concatenate([covidTargets, normalTargets, opacityTargets, pneumoniaTargets])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.25)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the CNN architecture
model = keras.Sequential(
    [
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(100, 100, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(4, activation="softmax"),
    ]
)

# Compile the model with the appropriate loss function, optimizer, and metrics
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

import plotly.graph_objs as go
import plotly.io as pio

# Train the model and save the history object
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

# Evaluate the model on the test data after each epoch and save the results
test_loss = []
test_acc = []
for epoch in range(1, 11):
    loss, acc = model.evaluate(x_test, y_test)
    test_loss.append(loss)
    test_acc.append(acc)
    print(f'Test loss after epoch {epoch}: {loss:.4f}')
    print(f'Test accuracy after epoch {epoch}: {acc:.4f}')

# Create the accuracy plot
fig_accuracy = go.Figure()
fig_accuracy.add_trace(go.Scatter(x=history.epoch,
                                  y=history.history['accuracy'],
                                  mode='lines+markers',
                                  name='Training Accuracy'))
fig_accuracy.add_trace(go.Scatter(x=history.epoch,
                                  y=history.history['val_accuracy'],
                                  mode='lines+markers',
                                  name='Validation Accuracy'))
fig_accuracy.add_trace(go.Scatter(x=list(range(1, 11)),
                                  y=test_acc,
                                  mode='lines+markers',
                                  name='Test Accuracy'))

fig_accuracy.update_layout(title='Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')

# Create the loss plot
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=history.epoch,
                              y=history.history['loss'],
                              mode='lines+markers',
                              name='Training Loss'))
fig_loss.add_trace(go.Scatter(x=history.epoch,
                              y=history.history['val_loss'],
                              mode='lines+markers',
                              name='Validation Loss'))
fig_loss.add_trace(go.Scatter(x=list(range(1, 11)),
                              y=test_loss,
                              mode='lines+markers',
                              name='Test Loss'))

fig_loss.update_layout(title='Loss', xaxis_title='Epoch', yaxis_title='Loss')

# Save the plots as PNG images
pio.write_image(fig_accuracy, 'accuracy_plot.png')
pio.write_image(fig_loss, 'loss_plot.png')