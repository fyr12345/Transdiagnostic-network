# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 01:14:18 2024

@author: fyr
"""

import os
import numpy as np
import SimpleITK as sitk

# 加载NIfTI文件并转换为numpy数组
def load_nifti_files(files):
    maps = []
    for file in files:
        img = sitk.ReadImage(file)
        data = sitk.GetArrayFromImage(img)
        maps.append(data)
    return np.array(maps)

# Assuming that a list of NIfTI files for structural lesion network mapping is already available
maps_files  = [
   "xx1",'xx2','xx3','xx4'
]



maps_files = load_nifti_files(maps_files)
n_files, x, y, z = maps_files.shape


flattened_maps = maps_files.reshape(n_files, -1).T


def custom_pca_no_centering(maps, n_components):
    cov_matrix = np.dot(maps.T, maps) / maps.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    principal_components = np.dot(maps, selected_eigenvectors)
    return principal_components, sorted_eigenvalues[:n_components]

n_components = 4
pcs, variance = custom_pca_no_centering(flattened_maps, n_components)

# Reshaping principal component scores to their original shape
reconstructed_maps = pcs.T.reshape(n_components, x, y, z)

# Save principal component scores as NIfTI file
def save_nifti_maps(maps, template_file, output_prefix, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    template_img = sitk.ReadImage(template_file)
    for i in range(maps.shape[0]):
        new_img = sitk.GetImageFromArray(maps[i])
        new_img.CopyInformation(template_img)
        output_path = os.path.join(output_dir, f'{output_prefix}_pc{i+1}.nii')
        sitk.WriteImage(new_img, output_path)
        print(f"Saved PCA component {i+1} to {output_path}")

output_dir = r'xx'
save_nifti_maps(reconstructed_maps, maps_files[0], 'pca', output_dir)

variance_ratios = variance / np.sum(variance)
print(" PCA explained variance ratios:", variance_ratios)
