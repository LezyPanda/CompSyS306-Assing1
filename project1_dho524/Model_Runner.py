import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# paths for files
mlp_model_path = 'mlp_model.joblib'
mlp_data_path = 'mlp_test_data.joblib'

# Load files
mlp_model = joblib.load(mlp_model_path)
mlp_x_test, mlp_y_test = joblib.load(mlp_data_path)

# predict
mlp_predictions = mlp_model.predict(mlp_x_test)
mlp_predicted_labels = np.argmax(mlp_predictions, axis=1) 

# find metrics
mlp_accuracy = accuracy_score(mlp_y_test, mlp_predicted_labels)
mlp_precision = precision_score(mlp_y_test, mlp_predicted_labels, average='weighted')
mlp_recall = recall_score(mlp_y_test, mlp_predicted_labels, average='weighted')
mlp_f1 = f1_score(mlp_y_test, mlp_predicted_labels, average='weighted')
mlp_classification_report = classification_report(mlp_y_test, mlp_predicted_labels)

print(f'MLP Model - Accuracy: {mlp_accuracy}')
print(f'MLP Model - Precision: {mlp_precision}')
print(f'MLP Model - Recall: {mlp_recall}')
print(f'MLP Model - F1 Score: {mlp_f1}')
print(f'MLP Model - Classification Report: \n{mlp_classification_report}')

# paths for files
svm_model_path = 'svm_model.joblib'
svm_data_path = 'svm_test_data.joblib'

# Load data
svm_model = joblib.load(svm_model_path)
svm_x_test, svm_y_test = joblib.load(svm_data_path)

# predict
svm_predictions = svm_model.predict(svm_x_test)

# find
svm_accuracy = accuracy_score(svm_y_test, svm_predictions)
svm_precision = precision_score(svm_y_test, svm_predictions, average='weighted')
svm_recall = recall_score(svm_y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(svm_y_test, svm_predictions, average='weighted')
svm_classification_report = classification_report(svm_y_test, svm_predictions)

print(f'\nSVM Model - Accuracy: {svm_accuracy}')
print(f'SVM Model - Precision: {svm_precision}')
print(f'SVM Model - Recall: {svm_recall}')
print(f'SVM Model - F1 Score: {svm_f1}')
print(f'SVM Model - F1 Score: \n{svm_classification_report}')