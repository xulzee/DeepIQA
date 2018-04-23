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
        self.wl_subj = float(1e3)
        self.wl_l2 = float(5e-3)
        self.wr_tv = float(1e-2)
        self.ign = int(4)
        self.ign_scale = int(8)

        self.conv1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 2, 1),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 2, 1),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 2, 1)
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1)
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, 3, 1, 1)
            torch.nn.ReLU()
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(4, 1),
            torch.nn.ReLU()
        )

        self.sobel_y = Variable(torch.Tensor(
            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                     dtype='float32').reshape((1, 1, 3, 3))))
        self.sobel_x = Variable(torch.Tensor(
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                     dtype='float32').reshape((1, 1, 3, 3))))

    def forward(self, img_dis, e, e_ds4):
        img_dis = self.bn1(self.conv1_1(img_dis))
        img_dis = self.conv2_1(img_dis)

        e = self.bn2(self.conv1_2(e))
        e = self.conv2_2(e)

        e = torch.cat((img_dis, e), dim=1)

        e = self.bn3(self.conv3(e))
        e = self.bn4(self.conv4(e))
        e = self.bn5(self.conv5(e))
        sens_map = self.conv6(e)

        pred_map = sens_map * e_ds4

        pred_map = pred_map[:, :, self.ign:-self.ign, self.ign:-self.ign]
        feat_vec = torch.mean(torch.mean(pred_map, 3), 2)
        mos_p = self.fc1(feat_vec)
        mos_p = self.fc2(mos_p)
        return mos_p

    def sobel(self, x):
        y_grad = conv2d(x, self.sobel_y, stride=1, padding=0)
        x_grad = conv2d(x, self.sobel_x, stride=1, padding=0)
        return y_grad, x_grad

    def get_total_variation(self, x, beta=1.5):
        """
        Calculate total variation of the input.
        Arguments
            x: 4D tensor image. It must have 1 channel feauture
        """
        y_grad, x_grad = self.sobel(x)
        tv = torch.mean((y_grad ** 2 + x_grad ** 2) ** (beta / 2))
        return tv

    def shave_border(self, feat_map):
        return feat_map[:, :, self.ign_scale:-self.ign_scale, self.ign_scale:-self.ign_scale]
