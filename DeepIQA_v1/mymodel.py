import h5py
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, data_file):
        self.file = h5py.File(str(data_file), 'r')
        self.img_dis = self.file['img_dis'][:].astype(np.float32) / 255.
        self.img_dis = torch.FloatTensor(self.img_dis).permute(3, 2, 1, 0)
        print(self.img_dis.shape)
        self.img_ref = self.file['img_ref'][:].astype(np.float32) / 255.
        self.img_ref = torch.FloatTensor(self.img_ref).permute(3, 2, 1, 0)
        # simple normalization in[0,1]
        self.label = self.file['label'][:].astype(np.float32)
        self.label = torch.FloatTensor(self.label).permute(1, 0)
        print(self.label.shape)

    def __len__(self):
        return self.img_dis.shape[0]

    def __getitem__(self, idx):
        img_dis = self.img_dis[idx, :, :, :]
        img_ref = self.img_ref[idx, :, :, :]
        label = self.label[idx, :]
        sample = {'img_dis': img_dis, 'img_ref': img_ref, 'label': label}
        return sample


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # model_config
        self.ign = int(4)

        self.conv1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 2, 1),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 2, 1),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, 3, 1, 1),
            torch.nn.ReLU()
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1, 8),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(8, 1),
            torch.nn.ReLU()
        )

    def forward(self, img_dis, e, e_ds4):
        img_dis = self.conv1_1(img_dis)
        img_dis = self.conv2_1(img_dis)

        e = self.conv1_2(e)
        e = self.conv2_2(e)

        e = torch.cat((img_dis, e), dim=1)

        e = self.conv3(e)
        e = self.conv4(e)
        e = self.conv5(e)

        sens_map = self.conv6(e)
        pred_map = sens_map * e_ds4
        pred_map = pred_map[:, :, self.ign:-self.ign, self.ign:-self.ign]
        feat_vec = torch.mean(torch.mean(pred_map, 3), 2)
        mos_p = self.fc1(feat_vec)
        mos_p = self.fc2(mos_p)
        return mos_p, sens_map
