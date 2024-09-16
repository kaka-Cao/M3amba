import glob
from torch.utils import data
import numpy as np
import torch
from scipy.io import loadmat


def get_data_path(data_name):
    path_in = '' + data_name
    train_path = path_in + '/train'
    test_path = path_in + '/test'
    train_files = glob.glob(train_path + '/*.mat')
    test_files = glob.glob(test_path + '/*.mat')
    return train_files, test_files


class Dataset(data.Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, item):
        data = loadmat(self.files[item])
        hsi = data['hsi']
        lidar = data['lidar']
        label = data['label']

        mask = (label >= 0).astype(np.int8)
        label[label < 0] = 0
        if len(hsi.shape) == 2:
            hsi = hsi[..., None]
        if len(lidar.shape) == 2:
            lidar = lidar[..., None]
        # input_data = np.concatenate([hsi, lidar], axis=-1).transpose(2, 0, 1)

        hsi = torch.from_numpy(hsi.transpose(2, 0, 1)).type(torch.FloatTensor)
        lidar = torch.from_numpy(lidar.transpose(2, 0, 1)).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        label = torch.from_numpy(label).type(torch.FloatTensor).long()
        return hsi, lidar, mask, label

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    x, y = get_data_path('Houston2013')
    print(x)


