import quilt3
# import pandas as pd
import csv
import mahotas as mh
import os

# It is possible to view the details and get the csv file at that URL,
# https://open.quiltdata.com/b/allencell/packages/aics/pipeline_integrated_single_cell
# and we have separated the corresponding part of the csv data.
a = open("Allen_Cell_data.csv")
a = csv.reader(a)
list=[]
for obj in a:
    obj = str(obj).replace("['", "")
    obj = str(obj).replace("']", "")
    list.append(obj)

pkg = quilt3.Package.browse(
    "aics/pipeline_integrated_single_cell",
    registry="s3://allencell",
    top_hash="7fd488f05ec41968607c7263cb13b3e70812972a24e832ef6f72195bdd35f1b2",
)

for a in range(len(list)):
    i = list[a]
    try:
        if not os.path.exists("./data/" + i):
        # print("downloading..." + i)
            pkg[i].fetch("./data/cell_images_3d/")
    except:
        print(i + "wrong!")
        os.remove("./data/" + i)
        a = a-1

