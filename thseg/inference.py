from __future__ import division
import sys
import os
from tools.utils import read_image
import yimage
from tools.metrics import get_acc_v2
import numpy as np
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import tqdm
from collections import OrderedDict
import tools.transform as tr
from tools.dataloader import InferenceSegmentation
import tools
import torch
from networks.get_model import get_net
from tools.parse_config_yaml import parse_yaml
import torch.onnx
import pdb


def test(testloader, model):
    test_num = testloader.dataset.num_sample
    label_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    
    if os.path.exists(os.path.join(param_dict['pred_path'],'train')) is False:        
        os.mkdir(os.path.join(param_dict['pred_path'],'train'))

    
    with torch.no_grad():
        batch_num = 0
        for i, data in tqdm.tqdm(enumerate(testloader), ascii=True, desc="test step"):  # get data

            images, img_path, folder = data['image'], data['img_path'], data['folder']

            i += images.size()[0]
            images = images.cuda()
            outputs = model(images)
            pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])
            batch_num += images.size()[0]
            for kk in range(len(img_path)):
                cur_name = os.path.basename(img_path[kk])
                pred_sub = pred[kk, :, :] #512*512 masks
                #label_all[i] = read_image(gt_path[kk], 'gt')
                predict_all[i] = pred_sub
                
                
                yimage.io.write_image(
                    os.path.join(param_dict['pred_path'],folder[kk], cur_name),
                    pred_sub,
                    color_table=param_dict['color_table'])
        """ precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            param_dict['save_dir_model']) """


def load_model(model_path):
    input_bands = param_dict['input_bands']

    model = get_net(param_dict['model_name'], input_bands, param_dict['num_class'],
                    param_dict['img_size'], param_dict['pretrained_model'])

    model = torch.nn.DataParallel(model, device_ids=[0])
    state_dict = torch.load(model_path)['net']
    new_state_dict = OrderedDict()
    #for k, v in state_dict.items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    
    model.load_state_dict(state_dict)
    
    model.cuda()
    model.eval()
    return model


if __name__ == '__main__':
    if len(sys.argv) == 1:
        yaml_file = 'config.yaml'
    else:
        yaml_file = sys.argv[1]
    param_dict = parse_yaml(yaml_file)

    for kv in param_dict.items():
        print(kv)

    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['gpu_id']
    gpu_list = [i for i in range(len(param_dict['gpu_id'].split(',')))]
    gx = torch.cuda.device_count()
    print('useful gpu count is {}'.format(gx))

    model_path = os.path.join(param_dict['model_dir'], 'valiou_best.pth')

    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor()])  # data pocessing and data augumentation

    
    road_test = InferenceSegmentation(txt_path='./data/inference_list.txt', transform=composed_transforms_val)  # get data
    
    testloader = DataLoader(road_test, batch_size=param_dict['batch_size'], shuffle=False,
                           num_workers=param_dict['num_workers'], drop_last=False)  # define traindata

    model = load_model(model_path)

    test(testloader, model)
