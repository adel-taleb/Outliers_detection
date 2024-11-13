import os
import shutil
import gdown
import tarfile


# Google Drive file ID for the compressed models file
file_id = "10Ezttnntsf7eozmNg04JBiGwJs-n-1KN"
tar_filename = "all_results.tar.gz"
model_dir = "all_results"

# Create the models directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Download the .tar.gz file
print("Downloading models .tar.gz file...")
gdown.download(f"https://drive.google.com/uc?id={file_id}", tar_filename, quiet=False)

# Extract the .tar.gz file into the models directory
print(f"Extracting {tar_filename}...")
with tarfile.open(tar_filename, "r:gz") as tar:
    tar.extractall(path=model_dir)
print(f"Extracted files to {model_dir}.")

# Clean up by removing the downloaded .tar.gz file
os.remove(tar_filename)
print(f"Removed {tar_filename} after extraction.")
shutil.copytree('all_results', 'fast_api/all_results', dirs_exist_ok=True)
print("All models downloaded and extracted successfully.")