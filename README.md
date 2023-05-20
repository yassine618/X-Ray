
Chest X-ray images are commonly used in the med-
ical field to diagnose respiratory conditions such

as COVID-19, pneumonia, and lung opacity (Non-
COVID lung infection). However, interpreting these

images accurately can be challenging even for trained
professionals. This is where machine learning comes
in - by training a model to classify these images, we can

potentially improve the accuracy and efficiency of di-
agnosis. In this project, we aim to classify chest X-ray

images into four categories: COVID-19, normal, lung
opacity, and viral pneumonia. To accomplish this, we
will use convolutional neural networks (CNNs), which
are a type of deep learning model commonly used for

image classification tasks. CNNs are particularly well-
suited for image classification tasks because they can

automatically learn and extract relevant features from
images, making them more effective than traditional
machine learning models for this type of data.
3 Data Preparation
We used a publicly available dataset from Kaggle
[https://www.kaggle.com/datasets/tawsifurrahman/
covid19-radiography-database] that consists of 21165
chest X-ray images . The dataset is split into four
categories: COVID19 (3 616 images), normal
(10,192 images), lung opacity( 6012 images), and
viral pneumonia (1345 images). For the first Model
, data preparation involved loading the images from
the file system into memory and resizing them to a
fixed size of 100x100 pixels. The images were then
converted to arrays and normalized to have values
between 0 and 1. Finally, the data was split into
training and validation sets, with 20
for the second Model, data preparation was done
using a data generator, which loaded images on the fly
from the file system and preprocessed them using the
VGG16 preprocessor. The images were resized to a
fixed size of 224x224 pixels, which is the input size
required by the VGG16 model. The data generator also
performed data augmentation by randomly applying
horizontal flips and rotations to the images. The data
was split into training and validation sets using the
validation split parameter of the flow from directory
method.

The data preparation process for the second model is
more efficient than the first one because it loads images
on the fly instead of loading them all into memory at
once. This allows for the processing of larger datasets
that may not fit in memory. Additionally, data aug-
mentation helps to prevent overfitting and improves
the model’s ability to generalize to new data
