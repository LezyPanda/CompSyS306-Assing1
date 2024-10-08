import lfrobot
import joblib
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize
from jetbot import Robot, Camera, bgr8_to_jpeg
import ipywidgets
import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display
from PIL import Image
import io
import glob
import os

def preprocess(image):
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Flatten the image array and normalize
    flatten = image.flatten() / 255.0
    return flatten

def expected(model, image):
    # Reshape the image to add a batch dimension
    image = np.reshape(image, (1, -1))  # Add batch dimension
    prediction = model.predict(image)
    return prediction

from jupyter_clickable_image_widget import ClickableImageWidget

camera = Camera.instance()
camera_widget = ClickableImageWidget(width=120, height=120)
snapshot_widget = ipywidgets.Image(width=120, height=120)
traitlets.dlink((camera, 'value'), (camera_widget, 'value'), transform=bgr8_to_jpeg)

def crop_center(image, crop_width, crop_height):
    height, width, _ = image.shape
    start_x = width // 2 - crop_width // 2
    start_y = height // 2 - crop_height // 2
    return image[start_y:start_y + crop_height, start_x:start_x + crop_width]

def test_model(_, content, msg):
    if content['event'] == 'click':
        data = content['eventData']
        x = data['offsetX']
        y = data['offsetY']
        cropped_image = crop_center(camera.value, 120, 120)
        image = preprocess(cropped_image)
        prediction = expected(model, image)
        predicted_label = np.argmax(prediction)
        print(predicted_label)
    else:
        print("failed")
        
    # display saved snapshot
    snapshot = camera.value.copy()
    snapshot = cv2.circle(snapshot, (x, y), 8, (0, 255, 0), 3)
    snapshot_widget.value = bgr8_to_jpeg(snapshot)
    count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))

camera_widget.on_msg(test_model)

data_collection_widget = ipywidgets.VBox([
    ipywidgets.HBox([camera_widget, snapshot_widget])
])

display(data_collection_widget)