import h5py
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from myfunc import downsample_img, log_diff_fn, normalize_lowpass_subt
from mymodel import Model, MyDataset


def test():
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
        print(i)
        output_txt = output.data[0].cpu().numpy()
        label_txt = label.data[0].cpu().numpy()

        outputfile = open(
            '/home/xulzee/Documents/IQA/output/TID2013_cost/test_vr_result_155.txt', 'a+')
        outputfile.write(('{} {:.7f} {:.7f}'.format(
            i, output_txt[0], label_txt[0])) + '\r\n')

    outputfile.close()


use_gpu = torch.cuda.is_available()
model = Model()
print('Model structure:', model)


if use_gpu:
    model = model.cuda()

model_weights_file = '/home/xulzee/Documents/IQA/output/TID2013_cost/155-5.0233284-0.0043344param.pth'
model.load_state_dict(torch.load(model_weights_file))
print('load weights from', model_weights_file)

test_dataset = MyDataset(
    data_file='/media/xulzee/备份/IQA_dataset/vr_jpeg.h5')  # test datasets
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

if __name__ == '__main__':
    test()
