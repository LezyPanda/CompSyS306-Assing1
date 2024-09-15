import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Paths to the saved MLP model and test data
mlp_model_path = 'mlp_model.joblib'
mlp_data_path = 'mlp_test_data.joblib'

# Load the MLP model and test data
mlp_model = joblib.load(mlp_model_path)
mlp_x_test, mlp_y_test = joblib.load(mlp_data_path)

# Make predictions on the MLP test data
mlp_predictions = mlp_model.predict(mlp_x_test)

# Calculate additional metrics for MLP model
mlp_true_labels = np.argmax(mlp_y_test, axis=1)
mlp_predicted_labels = np.argmax(mlp_predictions, axis=1)

mlp_accuracy = accuracy_score(mlp_true_labels, mlp_predicted_labels)
mlp_precision = precision_score(mlp_true_labels, mlp_predicted_labels, average='weighted')
mlp_recall = recall_score(mlp_true_labels, mlp_predicted_labels, average='weighted')
mlp_f1 = f1_score(mlp_true_labels, mlp_predicted_labels, average='weighted')

print(f'MLP Model - Accuracy: {mlp_accuracy}')
print(f'MLP Model - Precision: {mlp_precision}')
print(f'MLP Model - Recall: {mlp_recall}')
print(f'MLP Model - F1 Score: {mlp_f1}')

# Paths to the saved SVM model and test data
svm_model_path = 'svm_model.joblib'
svm_data_path = 'svm_test_data.joblib'

# Load the SVM model and test data
svm_model = joblib.load(svm_model_path)
svm_x_test, svm_y_test = joblib.load(svm_data_path)

# Make predictions on the SVM test data
svm_predictions = svm_model.predict(svm_x_test)

# Calculate additional metrics for SVM model
svm_accuracy = accuracy_score(svm_y_test, svm_predictions)
svm_precision = precision_score(svm_y_test, svm_predictions, average='weighted')
svm_recall = recall_score(svm_y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(svm_y_test, svm_predictions, average='weighted')

print(f'SVM Model - Accuracy: {svm_accuracy}')
print(f'SVM Model - Precision: {svm_precision}')
print(f'SVM Model - Recall: {svm_recall}')
print(f'SVM Model - F1 Score: {svm_f1}')