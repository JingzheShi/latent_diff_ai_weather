#load the data and output their shapes

import numpy as np
import os

file_names = os.listdir('.')
for file_name in file_names:
    file_path = os.path.join('.', file_name)
    data = np.load(file_path, allow_pickle=True)
    print(file_name, data.shape)