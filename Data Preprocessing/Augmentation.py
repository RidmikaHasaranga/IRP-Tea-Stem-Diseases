import os
import cv2
from PIL import Image, ImageEnhance
import random
import numpy as np

dataset_path = "D:\IIT\Subjects\(4605)IRP\Devlo/DataSet"
augmented_data_path = "D:\IIT\Subjects\(4605)IRP\Devlo/Augmented_DataSet"

os.makedirs(augmented_data_path, exist_ok=True)

num_augment = 5

# Augmentation Parameters
# rotations = [-15, -10, 10, 15]
brightness = (0.8, 1.2)
blur_probablity = 0.1
contrast = (0.8, 1.2)


def augment(image):
    # horizontal flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Vertical flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # rotation
    # angle = random.choice(rotations)
    # image = image.rotate(angle)

    # brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(*brightness))

    # contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(*contrast))

    # blur
    if random.random() > blur_probablity:
        image = blur_region(image)

    return image


def blur_region(image):
    # convert image to cv2
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = image_cv.shape[:2]

    # define range
    x_start = random.randint(0, w // 2)
    y_start = random.randint(0, h // 2)
    x_end = random.randint(w // 2, w)
    y_end = random.randint(h // 2, h)

    blurred = cv2.GaussianBlur(image_cv[y_start:y_end, x_start:x_end], (15, 15), 0)
    image_cv[y_start:y_end, x_start:x_end] = blurred

    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))


for class_name in os.listdir(dataset_path):
    print(f"augmenting class:{class_name}")
    class_path = os.path.join(dataset_path, class_name)
    if os.path.exists(class_path):
        class_output_path = os.path.join(augmented_data_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            with Image.open(image_path) as img:
                for i in range(num_augment):
                    augmented_image = augment(img)
                    augmented_image_name = f"aug_{i}_{image_name}"
                    augmented_image.save(os.path.join(class_output_path, augmented_image_name))
    print(f"successfully augmented class {class_name}")

print("completed")