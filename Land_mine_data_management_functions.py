import os
import shutil
import glob

# Function to create folders for landmine types (5-class structure)
def create_landmine_folders_phase_3(dest_root, landmine_types):
    for landmine_type in landmine_types:
        landmine_folder_path = os.path.join(dest_root, landmine_type)
        if not os.path.exists(landmine_folder_path):
            os.makedirs(landmine_folder_path, exist_ok=True)

# Function to copy images to corresponding folders (5-class structure)
def copy_images_to_landmine_folders_phase_3(source_root, dest_root, landmine_types, date_folders):
    for landmine_type in landmine_types:
        for date_folder in date_folders:
            source_folder_path = os.path.join(source_root, date_folder, 'JPG', landmine_type)
            jpg_files = glob.glob(os.path.join(source_folder_path, '*.jpg'))
            
            dest_folder_path = os.path.join(dest_root, landmine_type)
            for i, jpg_file in enumerate(jpg_files):
                base_file_name = os.path.basename(jpg_file)
                new_file_name = f"{date_folder}_{i + 1}.jpg"
                dest_file_path = os.path.join(dest_folder_path, new_file_name)
                shutil.copy2(jpg_file, dest_file_path)

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
                        dest_file = os.path.join(target_class_path, f"{date_folder}_{file}")
                        shutil.copy(src_file, dest_file)


# Function to create folders for 2-class structure
def create_landmine_folders_phase_1(dest_root, landmine_types):
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
    for class_name in landmine_types.keys():
        class_folder_path = os.path.join(dest_root, class_name)
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)

# Function to copy images to corresponding folders (2-class structure)
def copy_images_to_landmine_folders_phase_1(source_root, dest_root, landmine_types, date_folders):
    for class_name, sub_classes in landmine_types.items():
        for sub_class in sub_classes:
            for date_folder in date_folders:  # Include date_folders in the path
                sub_class_path = os.path.join(source_root, date_folder, 'JPG', sub_class)
                target_class_path = os.path.join(dest_root, class_name)
                if os.path.exists(sub_class_path):
                    for file in os.listdir(sub_class_path):
                        src_file = os.path.join(sub_class_path, file)
                        dest_file = os.path.join(target_class_path, f"{date_folder}_{file}")
                        shutil.copy(src_file, dest_file)

