from PIL import Image
import os

# Define the path where the PNG files are located
path = r'C:\Users\ridmi\Downloads'

# Iterate over all the files in the folder
for file_name in os.listdir(path):
    if file_name.endswith('.png'):
        # Open the image
        img = Image.open(os.path.join(path, file_name))

        # Convert the image to RGB (JPEGs don't support transparency)
        rgb_img = img.convert('RGB')

        # Create the new file name
        new_file_name = file_name.replace('.png', '.jpg')

        # Save the image as JPG
        rgb_img.save(os.path.join(path, new_file_name), 'JPEG')

print("Conversion complete!")
