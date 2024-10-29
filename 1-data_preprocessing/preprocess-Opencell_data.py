import os
import numpy as np
import pylab
import mahotas as mh
import tifffile
import copy
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import zoom

from tqdm import tqdm

# p1 = '../Projects/segmentation/data_project/data_project1'  # 2d
# p2 = '../Projects/segmentation/data/data1'          # 3d
# save_path = '../Projects/segmentation/3Dseg'
p1 = 'G:/data_project'  # 2d
p2 = 'G:/data'          # 3d
save_path = 'G:/3Dsegblock_normalize'

def watershed_new(threshed):
    threshed1, n_label = mh.label(threshed)
    a = np.unique(threshed1)
    for i in a:
        v = threshed[threshed1 == i].size
        if v < 100:
            threshed[threshed1 == i] = 0

    distances = mh.stretch(mh.distance(threshed))

    Bc = np.ones((60, 60))
    maxima = mh.morph.regmax(distances, Bc=Bc)
    labeled, n_label = mh.label(maxima, Bc=Bc)
    surface = (distances.max() - distances)
    areas = mh.cwatershed(surface, labeled)
    areas *= threshed

    return labeled, areas

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

def findcenters(labeled, areas):
    l = np.unique(labeled)[1:]

    x_indices, y_indices = np.where(labeled != 0)
    coms = center_of_mass(labeled, labeled, range(1, len(l) + 1))
    centers = [(int(com[0]), int(com[1])) for com in coms]

    centroids_image = np.zeros_like(labeled)
    i = 0
    for i in range(len(centers)):
        x, y = centers[i]
        centroids_image[x, y] = l[i]

    borders = np.zeros(areas.shape, bool)
    borders[0, :] = 1
    borders[-1, :] = 1
    borders[:, 0] = 1
    borders[:, -1] = 1

    at_border = np.unique(areas[borders])
    for obj in at_border:
        areas[areas == obj] = 0
        centroids_image[centroids_image == obj] = 0

    a = np.unique(areas)
    for i in a:
        v = (areas == i).sum()
        if v < 600:
            centroids_image[centroids_image == i] = 0
            areas[areas == i] = 0

    # pylab.imshow(mh.overlay(areas, centroids_image))
    # pylab.show()

    return centroids_image

def normalize_image_intensity(image):
    image = image.transpose([1, 2, 3, 0])
    image = image.astype(np.float32)
    for i, channel in enumerate(image):
        max_value = np.max(image[i])
        min_value = np.min(image[i])
        image[i] = (image[i] - min_value) / (max_value - min_value)
        # print(np.max(image[i]))

    image = image.transpose([3, 0, 1, 2])

    return image

for path in tqdm(os.listdir(p1)[426:]):
    print(path)
    path_next = p1 + '/' + path
    # path_next = os.path.join(p1, path)
    # print(path_next)  # G:/testdatabase/data_project/AAMP_ENSG00000127837
    for i in os.listdir(path_next):
        # print(i)  # OC-FOV_AAMP_ENSG00000127837_CID001050_FID00013888_proj.tif
        # mask = SegImageVoronoi(path_next + '/' + i)
        gray = mh.imread(path_next + '/' + i)
        i = i.replace('proj', 'stack')  # OC-FOV_AAMP_ENSG00000127837_CID001050_FID00013888_stack.tif

        # print(p2 + '/' + path + '/' + i)
        # G:/testdatabase/data/AAMP_ENSG00000127837/OC-FOV_AAMP_ENSG00000127837_CID001050_FID00013888_stack.tif
        image3d = tifffile.imread(p2 + '/' + path + '/' + i)
        i = i.replace('.tif', '')  # OC-FOV_AAMP_ENSG00000127837_CID001050_FID00013888_stack
        save_path1 = save_path + '/' + path + '/' + i
        # print(save_path1)  # G:/3Dsegblock/AAMP_ENSG00000127837/OC-FOV_AAMP_ENSG00000127837_CID001050_FID00013888_stack
        if not os.path.isdir(save_path1):
            os.makedirs(save_path1)

        # gray = mh.imread('2d.tif')
        # image3d = tifffile.imread('3d.tif')
        # print(gray.shape)  # 600*600
        image3d = normalize_image_intensity(image3d.astype(np.int32))

        # print('image3d.dtype', image3d.dtype)  # uint16
        image3d = image3d64(image3d)
        # print(image3d.shape)  # (64, 2, 600, 600)
        # print('image3d.dtype', image3d.dtype)  # uint16

        gray = mh.gaussian_filter(gray, 1.)
        threshed = (gray > gray.mean())
        threshed = mh.close_holes(threshed, Bc=np.ones((5, 5)))
        labeled, areas = watershed_new(threshed)

        centroids_image = findcenters(labeled, areas)

        x_indices, y_indices = np.where(centroids_image != 0)

        target_height = 128
        target_width = 128

        scale_height = target_height / 200
        scale_width = target_width / 200
        numb = 0
        for j in range(len(y_indices)):
            x, y = x_indices[j], y_indices[j]

            width = 200
            half_width = int(width * 0.5)
            x_start = max(0, x - half_width)
            x_end = min(gray.shape[0], x + half_width)
            y_start = max(0, y - half_width)
            y_end = min(gray.shape[1], y + half_width)

            patch_height = x_end - x_start
            patch_width = y_end - y_start

            if patch_height < width:
                diff = width - patch_height
                if x_start == 0:
                    x_end += diff
                if x_end == 600:
                    x_start -= diff
            if patch_width < width:
                diff = width - patch_width
                if y_start == 0:
                    y_end += diff
                if y_end == 600:
                    y_start -= diff

            patch = image3d[:, :, x_start:x_end, y_start:y_end]
            resized_image3d = zoom(patch, (1, 1, scale_height, scale_width))
            resized_image = resized_image3d.transpose(1, 0, 2, 3)
            resized_image = resized_image.astype(np.float16)
            numb += 1
            tifffile.imsave(save_path1 + '/' + i + '_' + str(numb) + '.tif', resized_image)

