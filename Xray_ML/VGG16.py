import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Use a generator to load and preprocess images
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.1)

train_generator = datagen.flow_from_directory(
    "EIT_Dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    "EIT_Dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load the pre-trained VGG16 model
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional layers in the VGG16 model
for layer in vgg.layers:
    layer.trainable = False

# Create a new model on top of the VGG16 model
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(validation_generator, verbose=0)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

y_pred = model.predict(validation_generator)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = validation_generator.labels

cm = confusion_matrix(y_test_labels, y_pred_labels)

fig_cm = go.Figure(data=[go.Heatmap(z=cm, x=['relaxed', 'stretched', 'box', 'salem','ok','peace','thumb'], y=['relaxed', 'stretched', 'box', 'salem','ok','peace','thumb'], colorscale='Viridis')])
fig_cm.update_layout(title='Confusion Matrix', xaxis_title='Predicted Class', yaxis_title='True Class')
fig_cm.show()

print(classification_report(y_test_labels, y_pred_labels))


model_save_path = "EIT_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
