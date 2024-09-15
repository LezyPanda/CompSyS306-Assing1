import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn import svm
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score,precision_score
import pickle
import joblib

def load_and_process_images(data_dir, labels_csv):
    flat_data_arr = []
    target_arr = []

    labels_df = pd.read_csv(labels_csv)
    categories = labels_df['ClassId'].astype(str).tolist()

    for category in categories:
        print(f'loading... category : {category}')
        folder_path = os.path.join(data_dir, category)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img_array = imread(img_path)
            img_resized = resize(img_array, (28, 28, 3))  # Resize images to 28x28x3
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(categories.index(category))
        print(f'loaded category:{category} successfully')

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target

    return df

# Path to the pickle file
pickle_file = 'images_labels_svm.pkl'

# Check if the pickle file exists
if os.path.exists(pickle_file):
    # Load the images and labels from the pickle file
    with open(pickle_file, 'rb') as f:
        df = pickle.load(f)
    print("Loaded images and labels from pickle file.")
else:
    # Load and process images from the folders
    labels_csv = 'traffic signs class/labels.csv'
    data_dir = 'traffic signs class/myData'
    df = load_and_process_images(data_dir, labels_csv)
    
    # Save the images and labels using pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump(df, f)
    print("Processed and saved images and labels to pickle file.")

x = df.iloc[:, :-1]  # all the features
y = df.iloc[:, -1]   # the target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
print('Splitted Successfully')

 
model=svm.SVC(C=1, kernel="linear").fit(x_train, y_train)


model.score(x_test, y_test)
print('The accuracy of the model is:', model.score(x_test, y_test) * 100, '%')
y_pred = model.predict(x_test)
print('The accuracy score of the model is:', accuracy_score(y_test, y_pred))
print('The recall score of the model is:', recall_score(y_test, y_pred, average='weighted'))

# Save the test data and the model using joblib
joblib.dump((x_test, y_test), 'svm_test_data.joblib')
joblib.dump(model, 'svm_model.joblib')
print("Test data and model saved in joblib format.")