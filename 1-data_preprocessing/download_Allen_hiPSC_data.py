import pandas as pd
import quilt3
from pathlib import Path
import os
from tqdm import tqdm

# It is possible to view the details and get the csv file at that URL,
# https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset
# and we have separated the corresponding part of the csv data.

# meta_df = pkg["metadata.csv"]()
meta_df = pd.read_csv('Allen_hiPSC_data.csv')
# a quick look at what are the columns
print(meta_df.columns)

save_path = "H:/data_Allens13-15/crop_raw/"
file = meta_df['crop_raw']

# connect to quilt
pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", registry="s3://allencell")

# for i in file[:1]:
#     pkg[i].fetch(save_path)

# for a in range(len(file)):
for i in tqdm(range(130000, 150000)):  # This is for parallel downloading of multiple .py files
    image_path = file[i]
    try:
        if not os.path.exists("H:/data_Allens13-15/" + image_path):
        # print("downloading..." + i)
            pkg[image_path].fetch(save_path)
    except:
        print(image_path + "wrong!")
        if os.path.exists("H:/data_Allens13-15/" + image_path):
            os.remove("H:/data_Allens13-15/" + image_path)
            print('delete successfully!')
        i = i - 1