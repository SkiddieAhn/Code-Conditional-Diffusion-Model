import kagglehub
import os
import shutil

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution")

print("Path to dataset files:", path)

# Move to the data directory in the project
base_path = os.path.dirname(os.path.abspath(__file__))
dest_path = os.path.join(base_path, "data")
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

print(f"Moving folders from {path} to {dest_path}...")

# Robustly find 'train' and 'test' folders
found_train_test = False
for root, dirs, files in os.walk(path):
    if 'train' in dirs and 'test' in dirs:
        for folder in ['train', 'test']:
            s = os.path.join(root, folder)
            d = os.path.join(dest_path, folder)
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        found_train_test = True
        break

if not found_train_test:
    # Fallback to copy everything if train/test not found in a pair
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(dest_path, item)
        if os.path.isdir(s):
            if os.path.exists(d): shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

print("Dataset prepared in the 'data' directory.")
