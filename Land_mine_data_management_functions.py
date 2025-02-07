import os
import shutil
import glob


'Phase 1: Free Zone vs. Landmine'

# Function to create folders
def create_landmine_folders_phase_1(dest_root, landmine_types):
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
    for class_name in landmine_types.keys():
        class_folder_path = os.path.join(dest_root, class_name)
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)

# Function to copy images to corresponding folders
def copy_images_to_landmine_folders_phase_1(source_root, dest_root, landmine_types, date_folders):
    for class_name, sub_classes in landmine_types.items():
        for sub_class in sub_classes:
            for date_folder in date_folders:  # Include date_folders in the path
                sub_class_path = os.path.join(source_root, date_folder, 'JPG', sub_class)
                target_class_path = os.path.join(dest_root, class_name)
                
                if os.path.exists(sub_class_path):
                    for file in os.listdir(sub_class_path):
                        src_file = os.path.join(sub_class_path, file)
                        
                        # Ensure a unique filename
                        base_name, ext = os.path.splitext(file)
                        unique_name = f"{date_folder}_{base_name}{ext}"
                        dest_file = os.path.join(target_class_path, unique_name)

                        counter = 1
                        while os.path.exists(dest_file):
                            unique_name = f"{date_folder}_{base_name}_{counter}{ext}"
                            dest_file = os.path.join(target_class_path, unique_name)
                            counter += 1
                        
                        shutil.copy(src_file, dest_file)


'Phase 2: surface vs. deep landmine'
# Function to create folders for 3-class structure
def create_landmine_folders_phase_2(dest_root, landmine_types):
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
    for class_name in landmine_types.keys():
        class_folder_path = os.path.join(dest_root, class_name)
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)

# Function to copy images to corresponding folders (3-class structure)
def copy_images_to_landmine_folders_phase_2(source_root, dest_root, landmine_types, date_folders):
    for class_name, sub_classes in landmine_types.items():
        for sub_class in sub_classes:
            for date_folder in date_folders:  # Include date_folders in the path
                sub_class_path = os.path.join(source_root, date_folder, 'JPG', sub_class)
                target_class_path = os.path.join(dest_root, class_name)

                if os.path.exists(sub_class_path):
                    for file in os.listdir(sub_class_path):
                        src_file = os.path.join(sub_class_path, file)
                        
                        # Ensure a unique filename
                        base_name, ext = os.path.splitext(file)
                        unique_name = f"{date_folder}_{base_name}{ext}"
                        dest_file = os.path.join(target_class_path, unique_name)

                        counter = 1
                        while os.path.exists(dest_file):
                            unique_name = f"{date_folder}_{base_name}_{counter}{ext}"
                            dest_file = os.path.join(target_class_path, unique_name)
                            counter += 1
                        
                        shutil.copy(src_file, dest_file)



