from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pandas as pd
import os
import numpy as np
import cv2
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import plotly.express as px

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

covid_path = "COVID-19_Radiography_Dataset/COVID"
covidUrls = os.listdir(covid_path)
covidImages, covidTargets = loadImages(covid_path, covidUrls, 1)

normal_path = "COVID-19_Radiography_Dataset/Normal"
normalUrls = os.listdir(normal_path)
normalImages, normalTargets = loadImages(normal_path, normalUrls, 0)

opacity_path = "COVID-19_Radiography_Dataset/Lung_Opacity"
opacityUrls = os.listdir(opacity_path)
opacityImages, opacityTargets = loadImages(opacity_path, opacityUrls, 2)

pneumonia_path = "COVID-19_Radiography_Dataset/Viral Pneumonia"
pneumoniaUrls = os.listdir(pneumonia_path)
pneumoniaImages, pneumoniaTargets = loadImages(pneumonia_path, pneumoniaUrls, 3)

covidImages = np.asarray(covidImages)
normalImages = np.asarray(normalImages)
opacityImages = np.asarray(opacityImages)
pneumoniaImages = np.asarray(pneumoniaImages)

data = np.concatenate([covidImages, normalImages, opacityImages, pneumoniaImages])
targets = np.concatenate([covidTargets, normalTargets, opacityTargets, pneumoniaTargets])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.25)
model = keras.models.load_model("modelcnn.h5")
y_pred_probs = model.predict(x_test)
print(y_test[:10])
# Get the predicted class for each input image
y_pred = np.argmax(y_pred_probs, axis=1)

# Print the shape of the predicted class array
print(y_pred.shape)  

# Print the first 10 predicted classes and their corresponding true classes
print(y_pred[:10])

# Define a list of class labels for display purposes
class_labels = ["Normal", "COVID-19", " lung opacity", "viral pneumonia"]


fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for i in range(16):
    row = i // 4
    col = i % 4
    image = x_test[i]
    true_label = class_labels[y_test[i]]
    predicted_label = class_labels[y_pred[i]]
    axs[row, col].imshow(image)
    axs[row, col].set_title(f"True: {true_label}\nPredicted: {predicted_label}")
    axs[row, col].axis("off")
plt.tight_layout()
plt.show()

