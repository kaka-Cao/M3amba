import os
import cv2
import numpy as np
import torch
import matplotlib.patheffects as PathEffects
import tqdm
from torchvision import transforms
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import Dataset
import data_loader
from sklearn.metrics import *
from utils import config
from net import net_vmamba_v4
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
colors_all = ["#66C5CC", "#F6CF71", "#F89C74", "#DCB0F2", "#87C55F", "#9EB9F3", "#FE88B1",
              "#C9DB74", "#8BE0A4", "#7F3C8D", "#11A579", "#3969AC", "#E73F74", "#A5AA99",
              "#FF9DA7"]


def scatter(x, colors):

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=np.array(colors_all)[colors.astype(np.int_)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(num_class):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24, color=colors_all[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


if __name__ == '__main__':
    device = 'cuda'
    use_data = 'Houston2013'  #'Augsburg'   'MUFFL'  'Houston2013'
    config = config.configs[use_data]
    num_class = config.num_class
    # save_weights_path = './save_weights_' + use_data + '/weights.pth'
    save_weights_path = 'best_weights.pth'
    train_files, test_files = data_loader.get_data_path(use_data)

    model = net_vmamba_v4.VSSM(input_shape=(32, 32), in_chans_hsi=config.input_hsi_channel, in_chans_lidar=config.input_lidar_channel,
                               num_classes=config.num_class).to(device)
    key = model.load_state_dict(torch.load(save_weights_path, map_location=device)['model_dict'])
    print(key)
    model.eval()

    test_data_loader = DataLoader(Dataset(test_files), batch_size=1,
                                  num_workers=4, shuffle=False)
    labels, predicts = [], []
    features = []
    for x_hsi, x_lidar, mask, y in tqdm(test_data_loader):
        x_hsi = x_hsi.to(device)
        x_lidar = x_lidar.to(device)
        mask = mask.numpy().flatten()
        y = y.numpy().flatten()
        idx = np.where(mask > 0)
        y = y[idx]
        labels.extend(y)

        with torch.no_grad():
            *_, f = model(x_hsi, x_lidar)
        # pred = torch.softmax(pred, dim=1)
        # pred = pred.cpu().numpy()[0]
        # print(pred.shape, f.shape)
        # f = f.cpu().numpy()[0].reshape((384, -1)).transpose(1, 0)
        f = f.cpu().numpy()[0].reshape((num_class, -1)).transpose(1, 0)
        # pred = np.argmax(pred, 0).flatten()[idx]
        f = f[idx]
        features.extend(f)
        # predicts.extend(pred)
        # break
    features = np.array(features)
    print(features.shape)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=666)
    X_tsne = tsne.fit_transform(features)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    labels = np.array(labels)
    print(labels.max())
    scatter(X_norm, labels)

    plt.savefig('./visual_result/' + 'Houston2013_tsne.png')
    plt.show()
