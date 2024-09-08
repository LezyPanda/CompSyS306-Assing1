import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from PIL import Image
import pickle
import time
import random

def load_and_process_images(data_dir, num_classes=43, max_samples_per_class=500):
    images = []
    labels = []
    for folder in range(num_classes):
        folder_path = os.path.join(data_dir, str(folder))
        img_names = os.listdir(folder_path)
        if len(img_names) > max_samples_per_class:
            img_names = random.sample(img_names, max_samples_per_class)
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).resize((28, 28))  # Resize images to 28x28
            img_array = np.array(img)
            images.append(img_array)
            
            # The label is the folder name
            label = folder
            labels.append(label)
        print('folder of label', folder, 'images loaded. Number of samples:', len(img_names))
    return np.array(images), np.array(labels)

# Path to the pickle file
pickle_file = 'images_labels_svm.pkl'

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

# Flatten the images to 2D
num_samples, height, width, channels = images.shape
images = images.reshape((num_samples, height * width * channels))

# Standardize the data
scaler = StandardScaler()
images = scaler.fit_transform(images)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=0)

# Convert y_train and y_test to 1-dimensional arrays
y_train = np.argmax(to_categorical(y_train, num_classes=43), axis=1)
y_test = np.argmax(to_categorical(y_test, num_classes=43), axis=1)

param_grid = {'C': [0.1, 1, 10], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['linear', 'rbf']}
svc = svm.SVC(probability=True, max_iter=100)
print("The training of the model is started, please wait for while as it may take few minutes to complete")
model = GridSearchCV(svc, param_grid, scoring='accuracy', n_jobs=4)
model.fit(x_train, y_train)
print('The Model is trained well with the given images')
print(model.best_params_)

model.score(x_test, y_test)
print('The accuracy of the model is:', model.score(x_test, y_test) * 100, '%')
y_pred = model.predict(x_test)
print('The confusion matrix of the model is:', confusion_matrix(y_test, y_pred))
print('The classification report of the model is:', classification_report(y_test, y_pred))
print('The accuracy score of the model is:', accuracy_score(y_test, y_pred))
print('The recall score of the model is:', recall_score(y_test, y_pred, average='weighted'))