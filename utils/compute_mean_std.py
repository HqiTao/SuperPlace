import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

def compute_mean_and_variance(database_folder):
    database_paths = sorted(glob(os.path.join(database_folder, "**", "*.jpg"), recursive=True))
    
    sum_image = None
    sum_square_image = None
    n_images = len(database_paths)

    new_size = (224, 224)

    for image_path in tqdm(database_paths):
        with Image.open(image_path) as img:
            img_resized = img.resize(new_size)
            img_array = np.array(img_resized) / 255.0
            if sum_image is None:
                sum_image = np.zeros_like(img_array)
                sum_square_image = np.zeros_like(img_array)
            
            sum_image += img_array
            sum_square_image += img_array ** 2

    mean = np.mean(sum_image / n_images, axis=(0, 1))
    std = np.sqrt(np.mean(sum_square_image / n_images - (sum_image / n_images) ** 2, axis=(0, 1)))

    return mean, std

database_folder = '/mnt/sda3/Projects/npr/datasets/gsv_cities/Images'
mean, variance = compute_mean_and_variance(database_folder)
print("Mean:", mean)
print("Variance:", variance)