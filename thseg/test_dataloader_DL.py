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
from tools.dataloader import IsprsSegmentation, AmodalSegmentation
import tools
import torch
from networks.get_model import get_net
from tools.parse_config_yaml import parse_yaml
import torch.onnx
import pdb
import networkDL as network
import torch.nn as nn

def test(testloader, model):
    test_num = testloader.dataset.num_sample
    label_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    predict_all_coarse = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    label_all_occ = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all_occ = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    
    if os.path.exists(param_dict['pred_path']) is False:
        os.mkdir(param_dict['pred_path'])
    
    if param_dict['prob'] == True:
        if os.path.exists(param_dict['pred_prob_path']) is False:
            os.mkdir(param_dict['pred_prob_path'])

    if os.path.exists(os.path.join(param_dict['pred_path'], 'coarse')) is False:
        os.makedirs(os.path.join(param_dict['pred_path'], 'coarse'))
    if os.path.exists(os.path.join(param_dict['pred_path'], 'occ')) is False:
        os.makedirs(os.path.join(param_dict['pred_path'], 'occ'))

    with torch.no_grad():
        batch_num = 0
        n = 0
        for i, data in tqdm.tqdm(enumerate(testloader), ascii=True, desc="test step"):  # get data

            if param_dict['amodal'] == True: 
                images, labels, img_path, gt_path, modal = data['image'], data['gt'], data['img_path'], data['gt_path'], data['visible']
                images = torch.cat((images,modal), dim = 1)
            else:
                #images, labels, img_path, gt_path, gt_path_mask, occ_mask = data['image'], data['gt'], data['img_path'], data['gt_path'], data['modal_path'], data['visible']
                images, labels, img_path, gt_path = data['image'], data['gt'], data['img_path'], data['gt_path']

            i += images.size()[0]
            images = images.cuda()
            #occ, coarse, outputs, visi = model(images)
            #occ, outputs = model(images)
            outputs = model(images)
            
            if param_dict['prob'] == False:
                #pred_occ = tools.utils.out2pred(occ, param_dict['num_class'], param_dict['thread'])
                #pred_coarse = tools.utils.out2pred(coarse, param_dict['num_class'], param_dict['thread'])
                pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])
            else: 
                pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'], prob= True)

            batch_num += images.size()[0]
            for kk in range(len(img_path)):
                cur_name = os.path.basename(img_path[kk])
                
                pred_sub = pred[kk, :, :]
                label_all[n] = read_image(gt_path[kk], 'gt')
                predict_all[n] = pred_sub
                
                n += 1
                #pred_sub_coarse = pred_coarse[kk, :, :]
                #predict_all_coarse[i] = pred_sub_coarse

                #pred_sub_occ = pred_occ[kk, :, :]
                #label_all_occ[i] = read_image(gt_path_mask[kk], 'gt')
                #predict_all_occ[i] = pred_sub_occ

                if param_dict['prob'] == False: 
                    yimage.io.write_image(
                        os.path.join(param_dict['pred_path'], cur_name.replace('.jpg','.png')),
                        pred_sub,
                        color_table=param_dict['color_table'])
                    
                    
                        

                    #yimage.io.write_image(
                    #    os.path.join(param_dict['pred_path'], 'coarse', cur_name.replace('.jpg','.png')),
                    #    pred_sub_coarse,
                    #    color_table=param_dict['color_table'])  

                    #yimage.io.write_image(
                    #    os.path.join(param_dict['pred_path'], 'occ', cur_name.replace('.jpg','.png')),
                    #    pred_sub_occ,
                    #    color_table=param_dict['color_table'])                  


                else:
                    
                    yimage.io.write_image(
                        os.path.join(param_dict['pred_prob_path'], cur_name),
                        pred_sub)
                        

        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            param_dict['save_dir_model'])
        #precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
        #    label_all, predict_all_coarse,
        #    param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
        #    os.path.join(param_dict['pred_path'], 'coarse'))
        #precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
        #    label_all_occ, predict_all_occ,
        #    param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
        #    os.path.join(param_dict['pred_path'], 'occ'))


def load_model(model_path):
    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=8) #8 or 16??? deeplabv3plus_mobilenet
    #network.convert_to_separable_conv(model.classifier)
    #set_bn_momentum(model.backbone, momentum=0.01)

    num_classes = param_dict['num_class']
    last_layer = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) 
        
    model.classifier.classifier[3] = last_layer    

    #model = get_net(param_dict['model_name'], input_bands, param_dict['num_class'],
    #                param_dict['img_size'], param_dict['pretrained_model'])

    #model = torch.nn.DataParallel(model, device_ids=gpu_list)
    model = torch.nn.DataParallel(model, device_ids=[0])
    
    #pdb.set_trace()
    state_dict = torch.load(model_path)['net']
    new_state_dict = OrderedDict()
    #for k, v in state_dict.items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    #pdb.set_trace()
    model.load_state_dict(state_dict)
    print('epoch: ', torch.load(model_path)['epoch'])
    #pdb.set_trace()
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

    model_path = os.path.join(param_dict['model_dir'], '290valiou_best.pth')#'valiou_best.pth')#0valiou_best.pth')220

    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor()])  # data pocessing and data augumentation

    if param_dict['amodal'] == False:
        road_test = IsprsSegmentation(txt_path=param_dict['test_list'], transform=composed_transforms_val)  # get data
    else:
        road_test = AmodalSegmentation(txt_path=param_dict['test_list'], transform=composed_transforms_val)  # get data
    
    testloader = DataLoader(road_test, batch_size=param_dict['batch_size'], shuffle=False,
                           num_workers=param_dict['num_workers'], drop_last=False)  # define traindata

    model = load_model(model_path)


    test(testloader, model)
