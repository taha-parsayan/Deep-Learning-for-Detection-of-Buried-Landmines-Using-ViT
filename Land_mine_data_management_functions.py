'''
Author: Taha Parsayan
Date: 2025 Jan 07
'''

import os
import pandas as pd
import shutil
import glob


# Make folders for each landmine type in 'data'
def create_landmine_folders(dest_root, landmine_types):
    # Create the corresponding folders in 'data'
    if not os.path.exists('data/train'):
        os.mkdir('data/train')
    if not os.path.exists('data/test'):
        os.mkdir('data/test')
    
    for landmine_type in landmine_types:
        landmine_folder_path = os.path.join(dest_root, landmine_type)
        if not os.path.exists(landmine_folder_path):
            os.makedirs(landmine_folder_path, exist_ok=True)


# Copy the images to the corresponding folders
def copy_images_to_landmine_folders(source_root, dest_root, landmine_types, date_folders):
    # Iterate through landmine types
    for landmine_type in landmine_types:
        # Iterate through date folders
        for date_folder in date_folders:
            # Construct the source folder path
            source_folder_path = os.path.join(source_root, date_folder, 'JPG', landmine_type)
            # Get all .jpg files in the source folder
            jpg_files = glob.glob(os.path.join(source_folder_path, '*.jpg'))
            
            # Define the destination folder path
            dest_folder_path = os.path.join(dest_root, landmine_type)
            
            # Copy each .jpg file to the destination folder
            for i, jpg_file in enumerate(jpg_files):
                # Extract the base filename (without the path)
                base_file_name = os.path.basename(jpg_file)
                
                # Add the date to the beginning of the file name
                new_file_name = f"{date_folder}_{i + 1}.jpg"
                
                # Define the destination file path
                dest_file_path = os.path.join(dest_folder_path, new_file_name)
                
                # Copy the file
                shutil.copy2(jpg_file, dest_file_path)