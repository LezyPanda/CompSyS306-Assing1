import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import csv
import torch


# Load the CSV file containing labels
file = 'traffic signs class/labels.csv'
data_dir = 'traffic signs class/myData'

def parse_csv(file_path):
    keys = []
    data_dict = {}
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Skip the header row
        next(reader)
        
        for row in reader:
            if len(row) == 2:
                key, name = row
                key = int(key)  # Convert the key to integer
                keys.append(key)
                data_dict[key] = name.strip()  # Remove any extra whitespace
    
    return keys, data_dict

labels, data_dict = parse_csv(file)

# Load the dataset using image_dataset_from_directory
train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(32, 32),
    batch_size=24,
    label_mode='int',
    validation_split=0.2,
    subset='both',
    seed=7414
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Normalize the images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Define the model
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),  # Adjust input shape to match the image size
    Dense(256, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(43, activation='softmax')  # Adjust output layer to match the number of classes
])

model.summary()
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=False),  # Use SparseCategoricalCrossentropy for integer labels
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(val_dataset, verbose=2)

print('\nTest accuracy:', test_acc)