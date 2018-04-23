import torch
import numpy as np
from .myfunc import normalize_lowpass_subt, downsample_img, log_diff_fn
from .mymodel import Model, MyDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()


def checkpoint(epoch, loss, test_mse):
    model.eval()
    if use_gpu:
        model.cpu()  # you should save weights on cpu not on gpu

    # save weights
    model_path = checkpoint_dir + '{}-{:.7f}-{:.7f}param.pth'.format(epoch, loss, test_mse)
    torch.save(model.state_dict(), model_path)

    # print and save record
    print('Epoch {} : loss:{:.7f} test_mse:{:.7f}'.format(epoch, loss, test_mse))
    print("Checkpoint saved to {}".format(model_path))

    output = open(checkpoint_dir + 'train_result.txt', 'a+')
    output.write(('{} {:.7f} {:.7f}'.format(epoch, loss, test_mse)) + '\r\n')
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

        model.eval()
        for i, sample in enumerate(test_dataloader):
            img_dis, img_ref, label = sample['img_dis'], sample['img_ref'], sample['label']
            e = log_diff_fn(img_ref, img_dis)
            e = Variable(e.cuda())
            e_ds4 = downsample_img(downsample_img(e)).cuda()
            img_dis = Variable(img_dis).cuda()
            label = Variable(label.cuda())
            img_dis_norm = normalize_lowpass_subt(img_dis)
            output = model(img_dis_norm, e, e_ds4)
            test_pred[i] = output.data[0].cpu().numpy()
            test_label[i] = label.data[0].cpu().numpy()

        test_mse = mse(test_pred, test_label)

        checkpoint(epoch, sum_loss / len(dataloader), test_mse)


if __name__ == '__main__':
    # Hyper Parameters
    EPOCH = 80
    BATCH_SIZE = 5
    LR = 0.0001

    use_gpu = torch.cuda.is_available()
    checkpoint_dir = '/home/xulzee/Documents/IQA/output/TID2013/'
    print('checkpoint dir :', checkpoint_dir)
    dataset = MyDataset(data_file='/home/xulzee/Documents/IQA/dataset/TID2013/train_live_iqa.h5')  # train datasets
    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataset = MyDataset(data_file='/home/xulzee/Documents/IQA/dataset/test_live_iqa.h5')  # test datasets
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Model()
    print('Model structure:', model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()

    if use_gpu:
        model = model.cuda()
        loss_func = loss_func.cuda()

    params = list(model.parameters())

    for i in range(len(params)):
        print('layer:', i + 1, params[i].size())

    print('length of dataset:', len(dataset))

    test_pred = np.zeros(len(test_dataset), dtype='float32')
    test_label = np.zeros(len(test_dataset), dtype='float32')
    train()
