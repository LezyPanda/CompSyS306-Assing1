import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import keras
from keras import Sequential
import joblib
import matplotlib.pyplot as plt
import pickle

# Function to load and process images
def load_and_process_images(data_dir, labels_csv):
    flat_data_arr = []
    target_arr = []

    labels_df = pd.read_csv(labels_csv)
    categories = labels_df['ClassId'].astype(str).tolist()
    # go though all the images and catagories
    for category in categories:
        print(f'loading... category : {category}')
        folder_path = os.path.join(data_dir, category)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img_array = imread(img_path)
            img_gray = rgb2gray(img_array)  # Convert to grayscale
            img_resized = resize(img_gray, (28, 28))  # Resize images to 28x28
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(categories.index(category))
        print(f'loaded category:{category} successfully')

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target

    return df

pickle_file = 'images_labels_processed.pkl'

# Check if the pickle file exists
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        df = pickle.load(f)
    print("Loaded images.")
else:
    # Load and process images from the folders
    labels_csv = 'traffic signs class/labels.csv'
    data_dir = 'traffic signs class/myData'
    df = load_and_process_images(data_dir, labels_csv)
    
    # Save the images and labels
    with open(pickle_file, 'wb') as f:
        pickle.dump(df, f)
    print("Saved images")

x = df.iloc[:, :-1]  
y = df.iloc[:, -1]  

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Define the model
model = Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(43, activation='softmax')  
])
model.summary()
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Lists to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
times = []

# Train the model
for epoch in range(10):
    start_time = time.time()
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=1)
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
epochs = range(1, 11)

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
test_images = x_test[:num_images_to_show].to_numpy()  
true_labels = y_test[:num_images_to_show].to_numpy()  
predicted_labels = np.argmax(model.predict(test_images), axis=1)

plt.figure(figsize=(10, 10))
for i in range(num_images_to_show):
    plt.subplot(1, num_images_to_show, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')  
    plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
    plt.axis('off')
plt.show()

# Save the test data and model
joblib.dump((x_test, y_test), 'mlp_test_data.joblib')
joblib.dump(model, 'mlp_model.joblib')
print("Test data and model saved in joblib format.")