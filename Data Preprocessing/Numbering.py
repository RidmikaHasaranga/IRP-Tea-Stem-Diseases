import os

# Define the path to the dataset
base_path = r'D:\IIT\Subjects\(4605)IRP\Devlo\DataSet'

# List of folder names to process
folders = ['Healthy', 'Live_Wood', 'Pink_Wax', 'Stem_Canker']


# Function to number the images in each folder
def number_images_in_folder(folder_path):
    # List all files in the folder
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    # Loop through and rename the files
    for i, filename in enumerate(images, start=1):
        # Get the file extension
        file_ext = os.path.splitext(filename)[1]

        # Construct the new filename with numbering (e.g., 1.jpg, 2.png)
        new_filename = f"{i}{file_ext}"

        # Full paths for renaming
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(src, dst)
        print(f"Renamed {filename} to {new_filename}")


# Loop through each folder and number the images
for folder in folders:
    folder_path = os.path.join(base_path, folder)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder} does not exist. Skipping.")
        continue

    print(f"Numbering images in {folder} folder...")
    number_images_in_folder(folder_path)

print("Image numbering complete.")
