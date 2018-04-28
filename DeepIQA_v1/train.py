import torch
import numpy as np
from myfunc import normalize_lowpass_subt, downsample_img, log_diff_fn
from mymodel import Model, MyDataset
from torch.nn.functional import conv2d
from torch.autograd import Variable
from torch.utils.data import DataLoader
from logger import Logger

sobel_y = Variable(torch.Tensor(
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
             dtype='float32').reshape((1, 1, 3, 3))))
sobel_x = Variable(torch.Tensor(
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             dtype='float32').reshape((1, 1, 3, 3))))


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()


def sobel(x):
    y_grad = conv2d(x, sobel_y.cuda(), stride=1, padding=0)
    x_grad = conv2d(x, sobel_x.cuda(), stride=1, padding=0)
    return y_grad, x_grad


def get_total_variation(x, beta=1.5):
    """
    Calculate total variation of the input.
    Arguments
        x: 4D tensor image. It must have 1 channel feauture
    """
    y_grad, x_grad = sobel(x)
    tv = torch.mean((y_grad ** 2 + x_grad ** 2) ** (beta / 2))
    return tv


def get_l2_regularization(net):
    reg_loss = None
    for name, param in net.named_parameters():
        if 'bias' not in name:
            if reg_loss is None:
                reg_loss = torch.sum((param ** 2))
            else:
                reg_loss += torch.sum((param ** 2))
    return reg_loss


def checkpoint(epoch, loss, subj_loss, l2_loss, tv_loss, test_mse):
    model.eval()
    if use_gpu:
        model.cpu()  # you should save weights on cpu not on gpu

    # save weights
    model_path = checkpoint_dir + \
        '{}-{:.7f}-{:.7f}param.pth'.format(epoch, loss, test_mse)
    torch.save(model.state_dict(), model_path)

    # print and save record
    print('Epoch {} : loss:{:.7f} test_mse:{:.7f}'.format(epoch, loss, test_mse))
    print("Checkpoint saved to {}".format(model_path))

    output = open(checkpoint_dir + 'train_result.txt', 'a+')
    output.write(('{} {:.7f} {:.7f} {:.7f} {:.7f} {:.7f}'.format(
        epoch, loss, subj_loss, l2_loss, tv_loss, test_mse)) + '\r\n')
    output.close()

    if use_gpu:
        model.cuda()  # don't forget return to gpu
    model.train()


def train():
    model.train()
    for epoch in range(39, EPOCH):
        sum_loss = 0.0
        sum_loss_subj = 0.0
        sum_loss_l2 = 0.0
        sum_loss_tv = 0.0
        model.train()
        # Train
        for iteration, sample in enumerate(dataloader):
            img_dis, img_ref, label = sample['img_dis'], sample['img_ref'], sample['label']

            e = log_diff_fn(img_ref, img_dis)
            e = Variable(e.cuda())
            e_ds4 = downsample_img(downsample_img(e)).cuda()
            img_dis = Variable(img_dis.cuda())
            label = Variable(label.cuda())

            img_dis_norm = normalize_lowpass_subt(img_dis)

            optimizer.zero_grad()
            output, sens_map = model.forward(img_dis_norm, e, e_ds4)
            # MSELoss
            subj_loss = loss_func(output, label)

            # L2 regularization
            l2_loss = get_l2_regularization(model)

            # Tv loss
            tv_loss = get_total_variation(sens_map, 3)

            loss = wl_subj * subj_loss + wl_l2 * l2_loss + wl_tv * tv_loss

            loss.backward()
            optimizer.step()

            if iteration % 1000 == 0:
                print("===> Epoch[{}]({}/{}): loss {:.7f} subj_loss {:.7f} l2_loss {:.7f} tv_loss {:.7f}".format(
                    epoch, iteration, len(dataloader),
                    loss.data[0], subj_loss.data[0], l2_loss.data[0], tv_loss.data[0]))
                print(torch.mean(output))
                info = {
                    'loss': loss.data[0],
                    'subj_loss': subj_loss.data[0],
                    'l2_loss': l2_loss.data[0],
                    'tv_loss': tv_loss.data[0],
                }

                for tag, value in info.items():
                    logger.scalar_summary(
                        tag, value, iteration + epoch * 23000)

            sum_loss += loss.data[0]
            sum_loss_subj += subj_loss.data[0]
            sum_loss_l2 += l2_loss.data[0]
            sum_loss_tv += tv_loss.data[0]
        # validation
        model.eval()
        for i, sample in enumerate(test_dataloader):
            img_dis, img_ref, label = sample['img_dis'], sample['img_ref'], sample['label']
            e = log_diff_fn(img_ref, img_dis)
            e = Variable(e.cuda())
            e_ds4 = downsample_img(downsample_img(e)).cuda()
            img_dis = Variable(img_dis).cuda()
            label = Variable(label.cuda())
            img_dis_norm = normalize_lowpass_subt(img_dis)
            output, sens_map = model(img_dis_norm, e, e_ds4)
            test_pred[i] = output.data[0].cpu().numpy()
            test_label[i] = label.data[0].cpu().numpy()

        test_mse = mse(test_pred, test_label)
        info = {
            'test_mse': test_mse
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        checkpoint(epoch, sum_loss / len(dataloader), sum_loss_subj / len(dataloader), sum_loss_l2 / len(dataloader),
                   sum_loss_tv / len(dataloader), test_mse)


if __name__ == '__main__':
    # Hyper Parameters
    EPOCH = 180
    BATCH_SIZE = 5
    LR = 1e-4

    wl_subj = float(1e3)
    wl_l2 = float(5e-3)
    wl_tv = float(1e-2)

    logger = Logger('./logs')
    use_gpu = torch.cuda.is_available()
    checkpoint_dir = '/home/xulzee/Documents/IQA/output/TID2013_cost_bn/'
    print('checkpoint dir :', checkpoint_dir)
    # Train dataset
    dataset = MyDataset(
        data_file='/media/xulzee/备份/IQA_dataset/TID2013/train_live_iqa.h5')  # train datasets
    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # Test dataset
    test_dataset = MyDataset(
        data_file='/media/xulzee/备份/IQA_dataset/TID2013/test_live_iqa.h5')  # test datasets
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = Model()
    print('Model structure:', model)
    model_weights_file = '/home/xulzee/Documents/IQA/output/TID2013_cost_bn/39-4.1797355-0.0033694param.pth'
    model.load_state_dict(torch.load(model_weights_file))
    print('load weights from', model_weights_file)
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
