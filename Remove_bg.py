import cv2
import numpy as np
import os

# Define the path to the dataset
base_path = r'D:\IIT\Subjects\(4605)IRP\Devlo\DataSet_Resized'

# List of folder names to process
folders = ['Healthy', 'Live_Wood', 'Pink_Wax', 'Stem_Canker']

# Define the path where processed images will be saved (without backgrounds)
output_base_path = r'D:\IIT\Subjects\(4605)IRP\Devlo\DataSet_No_Background'

# Ensure the output base path exists, create it if not
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)


# Function to remove background using GrabCut
def remove_background(image_path, output_image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Create a mask (same size as image, initialized to 0's)
    mask = np.zeros(img.shape[:2], np.uint8)

    # Create temporary arrays for GrabCut algorithm (initialized to 0's)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle that contains the object (foreground)
    height, width = img.shape[:2]
    rect = (10, 10, width - 20, height - 20)  # Adjust this if needed

    # Apply GrabCut algorithm
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Create a mask where 0 and 2 are background, 1 and 3 are foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    img_nobg = img * mask2[:, :, np.newaxis]

    # Save the result
    cv2.imwrite(output_image_path, img_nobg)


# Loop through each folder
for folder in folders:
    folder_path = os.path.join(base_path, folder)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder} does not exist. Skipping.")
        continue

    # Create the corresponding output folder
    output_folder_path = os.path.join(output_base_path, folder)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Loop through each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other image formats if needed
            image_path = os.path.join(folder_path, filename)
            output_image_path = os.path.join(output_folder_path, filename)

            print(f"Processing {image_path} and saving to {output_image_path}...")

            # Remove the background from the image
            remove_background(image_path, output_image_path)

print("Background removal and saving complete.")
