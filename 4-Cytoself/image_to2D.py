import numpy as np
import os
import pandas as pd
import tifffile
from tqdm import tqdm
data_path = '../Data/3Dsegblock_normalize'
path = pd.read_csv('../data_list/opencell_test_data.csv')['path']  
nuc = []
protein = []
for i in tqdm(path):
    new_path = i.replace("3DsegT", "../Data/3Dsegblock_normalize")
    image = tifffile.imread(new_path).astype(np.float32)  # (2,64,128,128)
    image_2d = np.max(image, axis=1)  # (2,128,128)

    nuc.append(image_2d[0])
    protein.append(image_2d[1])


nuc_array = np.stack(nuc, axis=0)
protein_array = np.stack(protein, axis=0)

np.save('../Data/my_data_2D/test_nuc.npy', nuc_array)
np.save('../Data/my_data_2D/test_pro.npy', protein_array)

