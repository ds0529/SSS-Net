import numpy as np
from os.path import join, exists, dirname, abspath
import os, sys, glob, pickle

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

ratio = 0.2
if ratio == 0.2:
    fs_num = 240
else:
    fs_num = 120
indices = np.random.choice(1201, fs_num, replace=False)
with open('utils/ScanNet_fs_samples_' + str(int(ratio*100)) + '.txt', 'w') as f:  # 设置文件对象
    for i in indices:
        file_name = 'train/' + str(i) + '.ply' + '\n'
        f.writelines(file_name)