import h5py
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d
from torch.utils.data import DataLoader, Dataset

from func import downsample_img, log_diff_fn, normalize_lowpass_subt

# import matplotlib.pyplot as plt

# Debug
# import cv2
# x_ = cv2.imread('img5.bmp', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('img', x_)
# cv2.waitKey(100)


# Hyper Parameters
EPOCH = 80
BATCH_SIZE = 5
LR = 0.0001

# prepare data


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
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU()
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(4, 1), torch.nn.ReLU()
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
        return feat_map[:, :, 8:-8, 8:-8]


def checkpoint(epoch, loss):
    model.eval()
    if use_gpu:
        model.cpu()  # you should save weights on cpu not on gpu

    # save weights
    model_path = checkpoint_dir + '{}-{:.7f}param.pth'.format(epoch, loss)
    torch.save(model.state_dict(), model_path)

    # print and save record
    print('Epoch {} : loss:{:.7f}'.format(epoch, loss))
    print("Checkpoint saved to {}".format(model_path))

    output = open(checkpoint_dir + 'train_result.txt', 'a+')
    output.write(('{} {:.7f}'.format(epoch, loss)) + '\r\n')
    output.close()

    if use_gpu:
        model.cuda()  # don't forget return to gpu
    model.train()


def train():
    model.train()

    for epoch in range(EPOCH):
        sum_loss = 0.0

        for iteration, sample in enumerate(dataloader):
            img_dis, img_ref, label = sample['img_dis'], sample['img_ref'], sample['label']

            e = log_diff_fn(img_ref, img_dis)
            e = Variable(e.cuda())
            e_ds4 = downsample_img(downsample_img(e)).cuda()
            img_dis = Variable(img_dis.cuda())
            label = Variable(label.cuda())

            # img_dis, img_ref, label = (Variable(img_dis.cuda()),Variable(img_ref.cuda()), Variable(label.cuda()))

            img_dis_norm = normalize_lowpass_subt(img_dis)

            optimizer.zero_grad()
            output = model.forward(img_dis_norm, e, e_ds4)

            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print("===> Epoch[{}]({}/{}): loss{:.7f}".format(epoch,
                                                                 iteration, len(dataloader), loss.data[0]))
            sum_loss += loss.data[0]

        checkpoint(epoch, sum_loss / len(dataloader))


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    checkpoint_dir = '/home/xulzee/Documents/IQA/output/TID2013/'

    print('checkpoint dir :', checkpoint_dir)

    dataset = MyDataset(
        data_file='/home/xulzee/Documents/IQA/dataset/TID2013/train_live_iqa.h5')  # train datasets
    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # test_dataset = MyDataset(data_file='/home/xulzee/Documents/IQA/dataset/test_live_iqa.h5')  # test datasets
    # test_dataloader = DataLoader(
    #     dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Model()
    print('Model structure:', model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()

    if use_gpu:
        model = model.cuda()
        loss_func = loss_func.cuda()

    params = list(model.parameters())
    # for i in range(len(params)):
    #    print('layer:', i + 1, params[i].size())

    print('length of dataset:', len(dataset))
    train()
