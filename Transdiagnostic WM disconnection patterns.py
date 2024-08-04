# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:49:46 2024

@author: fyr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:30:32 2024

@author: fyr
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy.stats import norm

# Path settings
nii_path = r'xx'
excel_path = r'xx'

#Clinical Information
df = pd.read_excel(excel_path)
y = df['group'].values

# Image Information
nii_files = [f for f in os.listdir(nii_path) if f.endswith('.nii')]
X = []

for nii_file in nii_files:
    img = sitk.ReadImage(os.path.join(nii_path, nii_file))
    data = sitk.GetArrayFromImage(img)
    X.append(data.flatten())

X = np.array(X)


scaler = StandardScaler()
X = scaler.fit_transform(X)


# Logistic Regression with 5-Fold Cross Validation and Grid Search for hyperparameter tuning
def logistic_regression_cv(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=60)
    y_true, y_pred = [], []
    all_weights = []

    param_grid = {'C': np.logspace(-5, 1, 10)}
    grid_search = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear'), param_grid, cv=kf, scoring='accuracy', n_jobs=20)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Store the weights
        all_weights.append(best_model.coef_.flatten())
        
        # Predict and store results for all test samples
        y_pred.extend(best_model.predict(X_test))
        y_true.extend(y_test)
    
    accuracy = accuracy_score(y_true, y_pred)
    mean_weights = np.mean(all_weights, axis=0)
    return accuracy, y_true, y_pred, mean_weights


# Running logistic regression for SDC data
accuracy_sdc, y_true_sdc, y_pred_sdc, mean_weights_sdc = logistic_regression_cv(X, y)


print("Accuracy SDC:", accuracy_sdc)
print("y_true_sdc length:", len(y_true_sdc))
print("y_pred_sdc length:", len(y_pred_sdc))



# Creating a new NIfTI image
def create_nifti_from_weights(mean_weights, reference_nii_path, output_nii_path):
    reference_img = sitk.ReadImage(reference_nii_path)
    reference_data = sitk.GetArrayFromImage(reference_img)
    reshaped_weights = mean_weights.reshape(reference_data.shape)
    

    new_img = sitk.GetImageFromArray(reshaped_weights)
    new_img.CopyInformation(reference_img)
    
    
    sitk.WriteImage(new_img, output_nii_path)

# Save mean_weights as a new NIfTI file
reference_nii_path = os.path.join(nii_path, nii_files[0])  
output_nii_path = r'xx'
create_nifti_from_weights(mean_weights_sdc, reference_nii_path, output_nii_path)


# Combined permutation test for feature significance and accuracy
def combined_permutation_test(X, y, mean_weights, original_accuracy, n_permutations=1000):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    param_grid = {'C': np.logspace(-5, 1, 10)}
    
    permuted_mean_weights = []
    permuted_accuracies = []

    for _ in range(n_permutations):
        y_permuted = shuffle(y, random_state=42)
        all_weights = []
        y_true_perm, y_pred_perm = [], []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_permuted[train_idx], y_permuted[test_idx]
            
            grid_search = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear'), param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Store the weights
            all_weights.append(best_model.coef_.flatten())
            
            # Predict and store results
            y_pred_perm.append(best_model.predict(X_test)[0])
            y_true_perm.append(y_test[0])
        
        permuted_mean_weights.append(np.mean(all_weights, axis=0))
        permuted_accuracies.append(accuracy_score(y_true_perm, y_pred_perm))
    
    permuted_mean_weights = np.array(permuted_mean_weights)
    permuted_accuracies = np.array(permuted_accuracies)
    
    p_values_features = np.mean(np.abs(permuted_mean_weights) >= np.abs(mean_weights), axis=0)
    p_value_accuracy = np.mean(permuted_accuracies >= original_accuracy)
    
    return p_values_features, p_value_accuracy

# Perform combined permutation test for feature significance and accuracy
p_values_sdc, accuracy_p_value = combined_permutation_test(X, y, mean_weights_sdc, accuracy_sdc)

# Print the permutation test results
print("Significant features p-values:", p_values_sdc)
print("Accuracy p-value:", accuracy_p_value)



# Create a new NIfTI image with non-significant feature weights set to 0
def create_nifti_from_weights(mean_weights, p_values, threshold, reference_nii_path, output_nii_path):
    significant_weights = mean_weights.copy()
    significant_weights[p_values >= threshold] = 0
    reference_img = sitk.ReadImage(reference_nii_path)
    reference_data = sitk.GetArrayFromImage(reference_img)
    reshaped_weights = significant_weights.reshape(reference_data.shape)
    

    new_img = sitk.GetImageFromArray(reshaped_weights)
    new_img.CopyInformation(reference_img)
    

    sitk.WriteImage(new_img, output_nii_path)


reference_nii_path = os.path.join(nii_path, nii_files[0])  
output_nii_path = r'xx'
create_nifti_from_weights(mean_weights_sdc, p_values_sdc, 0.001, reference_nii_path, output_nii_path)
