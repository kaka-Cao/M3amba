import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import Dataset
import data_loader
from net import net_vmamba_v4
from utils import config
from sklearn.metrics import *
import torch.nn as nn
import time
import clip
from PIL import Image
from torchvision.transforms import functional as F
from calflops import calculate_flops
from torchvision import models
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def aa(yt, j1t):
    yt = torch.from_numpy(np.array(yt))
    j1t = torch.from_numpy(np.array(j1t))
    mem = [0] * num_class
    all_mem = [0] * num_class
    n_r = [0] * num_class
    n_p = [0] * num_class
    oa = torch.eq(j1t, yt)

    for x, y in zip(yt, j1t):
        n_r[x] += 1
        n_p[y] += 1
        all_mem[x] += 1
        if x == y:
            # print(x, y )
            mem[x] += 1
    # print(mem)
    mem = np.array(mem)
    all_mem = np.array(all_mem) + 1e-3
    n_r = np.array(n_r)
    n_p = np.array(n_p)
    pe = np.sum(n_r * n_p) / (yt.shape[0] * yt.shape[0])
    acc_per_class = mem / all_mem
    # print(mem, all_mem)
    # print(acc_per_class)
    AA = np.mean(mem / all_mem)
    oa = torch.mean(oa.float())

    k = (oa - pe) / (1 - pe)
    # print(AA)
    return oa, AA, k, acc_per_class

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def get_loss(preds, target, mask, epoch):
    def loss_super(pred, target, mask):
        pred = pred.view(pred.shape[0], num_class, -1)
        target = target.view(target.shape[0], -1)
        mask = mask.view(mask.shape[0], -1)
        idxs = torch.where(mask > 0)
        pred = pred[idxs[0], :, idxs[1]]
        target = target[idxs]
        loss = criterion(pred, target).mean()
        return loss
    
    def loss_cosine(x, y, z):
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        z = z.reshape(z.shape[0], -1)
        target = torch.ones(x.shape[0]).to(device)
        loss = (criterion_cosine(x, z, target) + criterion_cosine(y, z, target)) / 2
        return loss

    hsi_pred, lidar_pred, pred_out = preds
    loss_hsi = loss_super(hsi_pred, target, mask)
    loss_lidar = loss_super(lidar_pred, target, mask)
    loss = loss_super(pred_out, target, mask)

    loss_cosine_ = loss_cosine(hsi_pred, lidar_pred, pred_out)

    rate = epoch / epochs + 1.
    rate1 = epoch / epochs
    
    return awl(rate * (loss + loss_hsi + loss_lidar), rate1 * loss_cosine_)

def print_memory_usage(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"[{tag}] Allocated memory: {allocated:.2f} MB")
        print(f"[{tag}] Reserved memory: {reserved:.2f} MB")

def fit():
    best_loss = 1000
    best_acc = 0
    for epoch in range(epochs):
        start_time = time.time()
        dt_size = len(train_data_loader.dataset)
        dt_size_val = len(test_data_loader.dataset)
        epoch_loss = 0
        step = 0
        pbar = tqdm(total=dt_size // batch_size,
                    desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict,
                    mininterval=0.3)
        for x_hsi, x_lidar, mask, y in train_data_loader:
            x_hsi = x_hsi.to(device)
            x_lidar = x_lidar.to(device)
            mask = mask.to(device)
            y = y.to(device)
            #================clip=================#
            x_hsi_clip = conv_hsi(x_hsi)
            x_lidar_clip = conv_lidar(x_lidar)
            x_hsi_clip = F.resize(x_hsi_clip, 224, interpolation=BICUBIC)
            x_lidar_clip = F.resize(x_lidar_clip, 224, interpolation=BICUBIC)
            #================clip=================#
            optimizer.zero_grad()
            #================clip=================#
            with torch.no_grad():
                hsi_image_features = model_CLIP.encode_image(x_hsi_clip)
                lidar_image_features = model_CLIP.encode_image(x_lidar_clip)
            #================clip=================#
            pred = model(x_hsi, x_lidar, hsi_image_features, lidar_image_features)

            loss = get_loss(pred, y, mask, epoch)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(**{
                'train_loss': epoch_loss / (step + 1)
            })
            pbar.update(1)
            step += 1
        pbar.close()
        end_time = time.time()
        one_epoch_train_time = end_time - start_time
        hours = int(one_epoch_train_time // 3600)
        minutes = int((one_epoch_train_time % 3600) // 60)
        seconds = int(one_epoch_train_time % 60)
        print(f"模型上一个epoch训练时间: {hours} 小时 {minutes} 分钟 {seconds} 秒")
        print_memory_usage("one_epoch_memory_use:") 
        epoch_loss_val = 0
        step_val = 0
        labels, predicts = [], []
        for x_hsi, x_lidar, mask, y in tqdm(test_data_loader):
            x_hsi = x_hsi.to(device)
            x_lidar = x_lidar.to(device)
            mask = mask.numpy().flatten()
            y = y.numpy().flatten()
            idx = np.where(mask > 0)
            y = y[idx]
            labels.extend(y)
            #================clip=================#
            x_hsi_clip = conv_hsi(x_hsi)
            x_lidar_clip = conv_lidar(x_lidar)
            x_hsi_clip = F.resize(x_hsi_clip, 224, interpolation=BICUBIC)
            x_lidar_clip = F.resize(x_lidar_clip, 224, interpolation=BICUBIC)
            with torch.no_grad():
                hsi_image_features = model_CLIP.encode_image(x_hsi_clip)
                lidar_image_features = model_CLIP.encode_image(x_lidar_clip)
            #================clip=================#
            with torch.no_grad():
                *_, pred = model(x_hsi, x_lidar, hsi_image_features, lidar_image_features)
            pred = torch.softmax(pred, dim=1)
            pred = pred.cpu().numpy()[0]
            pred = np.argmax(pred, 0).flatten()[idx]
            predicts.extend(pred)

        print(classification_report(labels, predicts))
        epoch_acc_dev, AA, k, acc_per_class = aa(labels, predicts)
        print(epoch_acc_dev.item(), AA.item(), k.item(), acc_per_class)
        if epoch !=0 and epoch % 50 == 0:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'optimizer_dict': optimizer.state_dict(),
                    'model_dict': model.state_dict(),
                    'conv_hsi': conv_hsi.state_dict(),
                    'conv_lidar': conv_lidar.state_dict(),
                }, save_weights_path + f'{str(epoch)}epoch.pth'
            )
        if best_acc < epoch_acc_dev.item():
            best_acc = epoch_acc_dev.item()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'optimizer_dict': optimizer.state_dict(),
                    'model_dict': model.state_dict(),
                    'conv_hsi': conv_hsi.state_dict(),
                    'conv_lidar': conv_lidar.state_dict(),
                }, save_weights_path + 'best_weights.pth'
            )
        print(f'best_OA: {best_acc}')
    print(best_acc)


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 8          
    use_data = 'MUFFL'        #'Augsburg'   'MUFFL'  'Houston2013'
    config = config.configs[use_data]
    num_class = config.num_class

    epochs = 200
    save_weights_path = './save_weights_' + use_data + '/'
    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
    train_files, test_files = data_loader.get_data_path(use_data)

    #========================================CLIP=============================================#
    model_CLIP, preprocess = clip.load("CLIP/ViT-B-32.pt", device=device)
    model_CLIP = model_CLIP.float()
    #========================================CLIP=============================================#

    model = net_vmamba_v4.VSSM(input_shape=(32, 32), in_chans_hsi=config.input_hsi_channel, in_chans_lidar=config.input_lidar_channel,
                               num_classes=config.num_class).to(device)

    conv_hsi = nn.Conv2d(config.input_hsi_channel, 3, 1, 1, 0).to(device)
    conv_lidar = nn.Conv2d(config.input_lidar_channel, 3, 1, 1, 0).to(device)

    optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': conv_hsi.parameters()}, {'params': conv_lidar.parameters()},],
                                  lr=1e-4, weight_decay=1e-3)  
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_cosine = nn.CosineEmbeddingLoss()
    awl = AutomaticWeightedLoss(2)

    train_data_loader = DataLoader(Dataset(train_files), batch_size=batch_size,
                                   num_workers=4, shuffle=True)
    test_data_loader = DataLoader(Dataset(test_files), batch_size=1,
                                  num_workers=4, shuffle=False)

    fit()
