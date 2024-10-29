import tifffile
import numpy as np
import mahotas as mh
import os

# p1 = 'G:/Allendata/data/test'  # 2d
p1 = 'G:/Allendata/data/cell_images_3d'  # 2d
# p2 = 'G:/testdatabase/data'          # 3d
save_path = 'G:/Allendata/data/allendata_2c'

for path in os.listdir(p1):
    print(path)  # AAMP_ENSG00000127837
    save_data = os.sep.join([save_path, path])
    # print(save_path1)
    # if not os.path.isdir(save_path1):
    #     os.makedirs(save_path1)
    if not os.path.exists(save_data):
        data_path = os.sep.join([p1, path])
        a = tifffile.imread(data_path)
        # print(a.dtype) #unit8. (6, 64, 168, 104)
        b = a.take([2, 4], 0)
        # print(b.shape)
        # tifffile.imwrite('new1.tif', b.astype(np.uint8))
        for i in range(64):
            a[0][i][a[0][i] > 0] = 1
            a[1][i][a[1][i] > 0] = 1
            b[0][i] = b[0][i] * a[0][i]
            b[1][i] = b[1][i] * a[1][i]

        images = np.pad(b, ((0, 0), (0, 0), (0, 0), (32, 32)))
        c = np.zeros([2, 64, 128, 128]).astype(np.uint8)
        for i3 in range(2):
            for j3 in range(64):
                c[i3][j3] = mh.resize_to(images[i3][j3], (128, 128))
        tifffile.imwrite(save_data, c.astype(np.uint8))

