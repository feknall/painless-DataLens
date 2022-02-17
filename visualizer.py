data_directory = '/home/hamid/PycharmProjects/DataLens/test1/samples/0.14epsilon-0.00delta.data'

import numpy as np

data = np.zeros((100000, 794))
dim = 0
import joblib
from tqdm import tqdm
for i in tqdm(range(1)):
    # x =  joblib.load(data_directory + f'-{i}.pkl')
    x = joblib.load(data_directory)
    data[dim: dim+len(x)] = x
    dim += len(x)

from matplotlib import pyplot as plt
for i in range(0, 100000, 10000):
  d = data[i, :-10]*256
  d = d.astype(np.uint16)
#   plt.imshow(data[i,:-10].reshape(28, 28), interpolation='nearest')
  plt.imshow(d.reshape(28, 28), cmap='gray')
  print(np.where(data[i, -10:] == 1))
  plt.show()


# import pandas as pd
#
# df_data = pd.DataFrame((data[:,:784]*256).astype(np.uint8))
#
# labels_one_hot = np.where(data[:, 784:] == 1)
# df_label = pd.DataFrame(labels_one_hot[1])
# df_data.to_csv('data-checkpoint-3-1.csv')
# df_label.to_csv('label-checkpoint-3-1.csv')