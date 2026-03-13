import kagglehub
import shutil
import os

print("Downloading dataset...")

path = kagglehub.dataset_download("arunavakrchakraborty/australia-weather-data")

print("Path to downloaded dataset files:", path)

dest_dir = "data"
os.makedirs(dest_dir, exist_ok=True)

print(f"Copying files to {dest_dir}...")
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(dest_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Dataset copied successfully!")
