import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import CategoricalCrossentropy

# Load the CSV file containing labels
data = 'traffic signs class/labels.csv'
df = pd.read_csv(data)

# Initialize lists to store images and labels
images = []
labels = []

# Load images from the folders and match them with their labels
data_dir = 'traffic signs class/myData'
for folder in range(43):
    folder_path = os.path.join(data_dir, str(folder))
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).resize((28, 28))  # Resize images to 28x28
        img_array = np.array(img)
        images.append(img_array)
        labels.append(folder)
    print('folder of label', folder, 'images loaded. Number of samples:', len(os.listdir(folder_path)))

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize the images
images = images / 255.0

# Convert labels to categorical
labels = to_categorical(labels, num_classes=43)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=0)

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28, 3)),  # Adjust input shape to include color channels
    Dense(256, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(43, activation='softmax')  # Adjust output layer to match the number of classes
])

model.summary()
model.compile(optimizer='adam',
              loss=CategoricalCrossentropy(from_logits=False),  # Use CategoricalCrossentropy for one-hot encoded labels
              metrics=['accuracy'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Train the model
model1 = model.fit(x_train, y_train, validation_split=0.2, epochs=20).to(device)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2).to(device)

print('\nTest accuracy:', test_acc)