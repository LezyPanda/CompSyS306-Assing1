import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from keras import Sequential
from keras.layers import Flatten, Dense, Input
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle

# Function to load and process images
def load_and_process_images(data_dir, num_classes=43):
    images = []
    labels = []
    for folder in range(num_classes):
        folder_path = os.path.join(data_dir, str(folder))
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).resize((28, 28))  # Resize images to 28x28
            img_array = np.array(img)
            images.append(img_array)
            
            # The label is the folder name
            label = folder
            labels.append(label)
        print('folder of label', folder, 'images loaded. Number of samples:', len(os.listdir(folder_path)))
    return np.array(images), np.array(labels)

# Path to the pickle file
pickle_file = 'images_labels.pkl'

# Check if the pickle file exists
if os.path.exists(pickle_file):
    # Load the images and labels from the pickle file
    with open(pickle_file, 'rb') as f:
        images, labels = pickle.load(f)
    print("Loaded images and labels from pickle file.")
else:
    # Load and process images from the folders
    data_dir = 'traffic signs class/myData'
    images, labels = load_and_process_images(data_dir)
    
    # Save the images and labels using pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump((images, labels), f)
    print("Processed and saved images and labels to pickle file.")

# Normalize the images
images = images / 255.0

# Convert labels to categorical
labels = to_categorical(labels, num_classes=43)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=0)

# Define the model
model = Sequential([
    Input(shape=(28, 28, 3)),  # Use Input layer as the first layer
    Flatten(),
    Dense(256, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(43, activation='softmax')  # Adjust output layer to match the number of classes
])
model.summary()
model.compile(optimizer='adam',
              loss=CategoricalCrossentropy(from_logits=False),  # Use CategoricalCrossentropy for one-hot encoded labels
              metrics=['accuracy'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lists to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
times = []

# Train the model
for epoch in range(2):
    start_time = time.time()
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=1, verbose=1)
    end_time = time.time()
    
    train_loss = history.history['loss'][0]
    val_loss = history.history['val_loss'][0]
    train_acc = history.history['accuracy'][0]
    val_acc = history.history['val_accuracy'][0]
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    times.append(end_time - start_time)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# Plotting the metrics
epochs = range(1, 3)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, 'bo-', label='Training loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, 'bo-', label='Training accuracy')
plt.plot(epochs, val_accuracies, 'ro-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, times, 'go-', label='Time per epoch')
plt.title('Time taken per epoch')
plt.xlabel('Epochs')
plt.ylabel('Time (seconds)')
plt.legend()

plt.tight_layout()
plt.show()

# Show a few test images with their predicted labels
num_images_to_show = 5
test_images = x_test[:num_images_to_show]
true_labels = np.argmax(y_test[:num_images_to_show], axis=1)
predicted_labels = np.argmax(model.predict(test_images), axis=1)

plt.figure(figsize=(10, 10))
for i in range(num_images_to_show):
    plt.subplot(1, num_images_to_show, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
    plt.axis('off')
plt.show()