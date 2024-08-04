# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 17:01:21 2024

@author: fyr
"""

import os
import numpy as np
import SimpleITK as sitk
from scipy.stats import pearsonr

# No threshold weights file
st = r'xx'  
st_img = sitk.ReadImage(st)
st_img = sitk.GetArrayFromImage(st_img)
st_data = st_img.flatten()  
# Anatomy of a blood vessel
additional_nifti_file1 = r'xxi'  
additional_img1 = sitk.ReadImage(additional_nifti_file1)
additional_data1 = sitk.GetArrayFromImage(additional_img1)
additional_data1 = additional_data1.flatten()  

# Only voxels that are not 0 in Anatomy of a blood vessel are retained.
nonzero_indices = np.where(additional_data1 != 0)
filtered_st_data = st_data[nonzero_indices] 
filtered_additional_data1 = additional_data1[nonzero_indices]

# Calculate the true spatial correlation
real_correlation = pearsonr(filtered_st_data, filtered_additional_data1)[0]



###null model test
##null model save file
nifti_folder = r'xx'  

# Anatomy of a blood vessel
additional_nifti_file = r'N:\STROKETOF\output_directory_Resliced\Resliced_3d_image_4f.nii'
additional_img = sitk.ReadImage(additional_nifti_file)
additional_data = sitk.GetArrayFromImage(additional_img)
additional_data = additional_data.flatten()  

#Comparison with each null model
def process_nifti_file(nifti_file, additional_data, real_correlation):
    img = sitk.ReadImage(nifti_file)
    data = sitk.GetArrayFromImage(img).flatten()
    
    
    nonzero_indices = np.where(additional_data != 0)
    filtered_data = data[nonzero_indices]
    filtered_additional_data = additional_data[nonzero_indices]
    
   
    correlation = pearsonr(filtered_data, filtered_additional_data)[0]
    
    return correlation > real_correlation, correlation

# Get each null model file
nifti_files = [os.path.join(nifti_folder, f) for f in os.listdir(nifti_folder) if f.endswith('.nii')]


num_greater = 0
correlation_values = []


for file in nifti_files:
    is_greater, correlation = process_nifti_file(file, additional_data, real_correlation)
     
    if correlation > abs(real_correlation):
        num_greater += 1
    correlation_values.append(correlation)


print(f"Number of correlations greater than real_correlation: {num_greater}")
print(f"Correlation values for each file: {correlation_values}")