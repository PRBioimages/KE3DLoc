import pandas as pd
import quilt3
from pathlib import Path
import os

import torch
from tqdm import tqdm
import tifffile
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import zoom
import zlib
def image3d64(image3d):
    if image3d.shape[0] < 64:
        n = 64 - image3d.shape[0]
        a = (np.floor(n / 2)).astype(int)
        b = (np.ceil(n / 2)).astype(int)
        # print(a, b)
        image3d = np.pad(image3d, ((a, b), (0, 0), (0, 0), (0, 0),))

    n = image3d.shape[0] + 1
    n = (np.floor(n / 2)).astype(int)
    a = n - 32
    b = n + 32
    image3d = image3d[a:b, :, :, :]

    return image3d

def normalize_image_intensity(image):
    image = image.transpose([1, 2, 3, 0])
    image = image.astype(np.float32)
    for i, channel in enumerate(image):
        max_value = np.max(image[i])
        min_value = np.min(image[i])
        image[i] = (image[i] - min_value) / (max_value - min_value)

    image = image.transpose([3, 0, 1, 2])

    return image

# meta_df = pkg["metadata.csv"]()
meta_df = pd.read_csv('metadata-crop.csv')
# a quick look at what are the columns
print(meta_df.columns)

save_path = "G:/data_Allens_normalize"
file_path = 'G:/data_Allens0-2'
image_file = meta_df['crop_raw']
mask_file = meta_df['crop_seg']

import warnings

warnings.simplefilter("always")

for i in tqdm(range(20000)):
    image_path = image_file[i]
    mask_path = mask_file[i]
    # for a in range(len(file)):
    if not os.path.exists(os.path.join(save_path, image_path)):
        if not os.path.exists(os.path.join(file_path, image_path)) or not os.path.exists(
                os.path.join(file_path, mask_path)):
            print(f"Error: One or both of the files {image_path} and {mask_path} do not exist.")
            continue
        try:
            image = tifffile.imread(os.path.join(file_path, image_path)).astype(np.float32)
        except (zlib.error, ValueError) as e:
            if os.path.exists(os.path.join(file_path, image_path)):
                os.remove(os.path.join(file_path, image_path))
            print(f"Error reading image: {e}. Retrying...")
            continue

        try:
            mask = tifffile.imread(os.path.join(file_path, mask_path)).astype(np.float32)
        except (zlib.error, ValueError) as e:
            if os.path.exists(os.path.join(file_path, mask_path)):
                os.remove(os.path.join(file_path, mask_path))
            print(f"Error reading mask: {e}. Retrying...")
            continue

        image_path = image_file[i]
        mask_path = mask_file[i]
        image = tifffile.imread(os.path.join(file_path, image_path)).astype(np.float32)
        mask = tifffile.imread(os.path.join(file_path, mask_path)).astype(np.float32)
        image = image[:, [0, 2], :, :]
        mask = mask[:, [0, 2], :, :]

        image3d = normalize_image_intensity(image)
        image3d = image3d * (mask / 255)
        image3d = image3d64(image3d)

        images = torch.tensor(image3d)
        wd = images.shape[2:]
        d = (max(wd) - min(wd)) / 2
        d = torch.tensor(d)
        a = torch.floor(d).type(dtype=torch.uint8) + 0
        b = torch.ceil(d).type(dtype=torch.uint8) + 0
        if wd[0] < wd[1]:
            images = F.pad(images, (0, 0, a, b))

        else:
            images = F.pad(images, (a, b, 0, 0))

        images = images.numpy()
        scale_height = 128 / images.shape[3]
        import warnings

        try:
            resized_image3d = zoom(images, (1, 1, scale_height, scale_height))
        except UserWarning as warning:
            print("Warning:", warning)
            print("Resized image shape:", resized_image3d.shape)
            if resized_image3d.shape != (64, 2, 128, 128):
                print("Warning: Resized image shape does not match expected shape. Actual shape:",
                      resized_image3d.shape)

        resized_image = resized_image3d.transpose(1, 0, 2, 3)
        resized_image = resized_image.astype(np.float16)
        tifffile.imsave(os.path.join(save_path, image_path), resized_image)

