import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

crop_size = 32


def nor(img):
    img -= np.min(img)
    img = img / np.max(img)
    return img


def crop_data(img1, img2, label, path_out_train, path_out_test, train=True):
    if len(img1.shape) == 2:
        img1 = np.pad(nor(img1), (crop_size // 2, crop_size // 2), constant_values=0)
    else:
        img1 = np.stack([np.pad(nor(img1[..., i]), (crop_size // 2, crop_size // 2), constant_values=0)
                         for i in range(img1.shape[-1])], axis=-1)
    if len(img2.shape) == 2:
        img2 = np.pad(nor(img2), (crop_size // 2, crop_size // 2), constant_values=0)
    else:
        img2 = np.stack([np.pad(nor(img2[..., i]), (crop_size // 2, crop_size // 2), constant_values=0)
                         for i in range(img2.shape[-1])], axis=-1)
    x, y = label.shape
    label = np.pad(label, (crop_size // 2, crop_size // 2), constant_values=0)
    l = len(os.listdir(path_out_train))
    ii = 0
    if train:
        for i in range(crop_size // 2, x + crop_size // 2, crop_size // 4):
            for j in range(crop_size // 2, y + crop_size // 2, crop_size // 4):
                if label[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2].sum() == 0:
                    continue
                savemat(path_out_train + '/' + str(ii) + '.mat', {
                    'hsi': img1[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2],
                    'lidar': img2[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2],
                    'label': label[i - crop_size // 2: i + crop_size // 2,
                             j - crop_size // 2: j + crop_size // 2].astype(
                        np.int8) - 1,
                })
                ii += 1
    else:
        for i in range(crop_size // 2, x + crop_size // 2, crop_size):
            for j in range(crop_size // 2, y + crop_size // 2, crop_size):
                if label[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2].sum() == 0:
                    continue
                savemat(path_out_test + '/' + str(ii) + '.mat', {
                    'hsi': img1[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2],
                    'lidar': img2[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2],
                    'label': label[i - crop_size // 2: i + crop_size // 2,
                             j - crop_size // 2: j + crop_size // 2].astype(
                        np.int8) - 1,
                })
                ii += 1


def crop_houston():
    hsi = loadmat('../data_begin/Houston2013/HSI.mat')['HSI']

    lidar = loadmat('../data_begin/Houston2013/LiDAR_1.mat')['LiDAR']
    train_label = loadmat('../data_begin/Houston2013/TRLabel.mat')[
        'TRLabel']
    test_label = loadmat('../data_begin/Houston2013/TSLabel.mat')[
        'TSLabel']

    print(train_label.min(), train_label.max())
    print(test_label.min(), test_label.max())

    path_out_train = '../data/Houston2013/train'
    os.makedirs(path_out_train, exist_ok=True)
    path_out_test = '../data/Houston2013/test'
    os.makedirs(path_out_test, exist_ok=True)
    crop_data(hsi, lidar, train_label, path_out_train, path_out_test)
    crop_data(hsi, lidar, test_label, path_out_train, path_out_test, False)


def crop_muufl():
    hsi = loadmat('../data_begin/MUFFL/hsi.mat')['HSI']

    lidar = loadmat('../data_begin/MUFFL/lidar.mat')['lidar']

    labels = loadmat('../data_begin/MUFFL/train_test_gt.mat')
    train_label = labels['trainlabels']
    test_label = labels['testlabels']

    print(train_label.min(), train_label.max())
    print(test_label.min(), test_label.max())
    path_out_train = '../data/MUFFL/train'
    os.makedirs(path_out_train, exist_ok=True)
    path_out_test = '../data/MUFFL/test'
    os.makedirs(path_out_test, exist_ok=True)
    crop_data(hsi, lidar, train_label, path_out_train, path_out_test)
    crop_data(hsi, lidar, test_label, path_out_train, path_out_test, False)


def crop_Augsburg():
    hsi = loadmat('../data_begin/Augsburg/data_hsi.mat')['data']

    lidar = loadmat('../data_begin/Augsburg/data_sar.mat')['data']
    train_label = loadmat('../data_begin/Augsburg/mask_train.mat')[
        'mask_train']
    test_label = loadmat('../data_begin/Augsburg/mask_test.mat')[
        'mask_test'].astype(np.uint8)

    print(train_label.min(), train_label.max())
    print(test_label.min(), test_label.max())

    path_out_train = '../data/Augsburg/train'
    os.makedirs(path_out_train, exist_ok=True)
    path_out_test = '../data/Augsburg/test'
    os.makedirs(path_out_test, exist_ok=True)
    crop_data(hsi, lidar, train_label, path_out_train, path_out_test)
    crop_data(hsi, lidar, test_label, path_out_train, path_out_test, False)


if __name__ == '__main__':
    # crop_muufl()
    # crop_Augsburg()
    # crop_Trento()
    crop_Augsburg()
    #
    # data = loadmat('../data_begin/Augsburg/data_DSM.mat')
    # print(data.keys())
    # key = list(data.keys())[-1]
    # print(data[key].shape)
    # print(data[key].max())
    #
    # plt.imshow(data[key].mean(-1))
    # plt.show()
