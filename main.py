import tensorflow as tf
import keras
from PIL import Image, ImageOps
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Define a custom DepthwiseConv2D that ignores 'groups' argument
class CustomDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']  # Remove unsupported argument
        super().__init__(*args, **kwargs)


# Load the model using the custom DepthwiseConv2D
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
model = tf.keras.models.load_model("keras_Model.h5", custom_objects=custom_objects, compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The len or number of images you can put into the array is determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Define the path to the image directory
# Change if you download from git for your own use
image_dir = r"C:\Users\andre\PycharmProjects\CloudClassification\test_images"

# Iterate over all files in the directory
for file_name in os.listdir(image_dir):
    file_path = os.path.join(image_dir, file_name)

    # Check if the file is an image
    # Add more if you need
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        try:
            # Open and process the image
            image = Image.open(file_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # Turn the image into a numpy array and normalize it
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predict using the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Print the prediction and confidence score
            print(f"File: {file_name}")
            print(f"Class: {class_name.strip()}, Confidence Score: {confidence_score}\n")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
