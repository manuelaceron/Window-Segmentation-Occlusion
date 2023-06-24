from __future__ import division
from tools.utils import read_image
import sys
import numpy as np
import os, time
import yimage
from tools.metrics import get_acc_info, get_acc_v2
from timm.optim import create_optimizer_v2
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import tqdm
import tools.transform as tr
from tools.dataloader import IsprsSegmentation, AmodalSegmentation
import tools
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from networks.get_model import get_net
from tools.losses import get_loss, get_weights
from tools.parse_config_yaml import parse_yaml
import torch.onnx
import pdb
import matplotlib.pyplot as plt
import cv2


def main():
      
    train_dataset = AmodalSegmentation(txt_path=param_dict['train_list'])  # get data
    val_dataset = AmodalSegmentation(txt_path=param_dict['val_list'])  # get data 
    
    trainloader = DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True,
                                num_workers=param_dict['num_workers'], drop_last=False)  # define traindata new: original was drop_las true

    valloader = DataLoader(val_dataset, batch_size=param_dict['batch_size'], shuffle=False,
                            num_workers=param_dict['num_workers'], drop_last=False)  # define traindata new: original was drop_las true                            


    
    image_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    n = 0
    for i, data in tqdm.tqdm(enumerate(trainloader)):  # get data
        output_list = []
        images, labels  = data['image'], data['gt'] #First batch of N images 3x5125x512
        images = images.cuda()
        labels = labels.cuda()
        

        for kk in range(len(images)):
            cur_name = os.path.basename(img_path[kk])
            pred_sub = images[kk, :, :]
            image_all[n] = images[kk]
            n += 1
        pdb.set_trace()         

def eval(valloader, model, criterion, epoch):
    
    val_num = valloader.dataset.num_sample
    label_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    model.eval()
    if param_dict['val_visual']:
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual')) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual'))
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch))) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)))
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice'))
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 't_slice'))
    if param_dict['bayesian']:
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'unct_val_visual')) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'unct_val_visual'))
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch))) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch)))
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch), 'slice'))
            
    with torch.no_grad():
        
        batch_num = 0
        val_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(valloader), ascii=True, desc="validate step"):  # get data
            output_list = []
            
            if param_dict['amodal'] == True: 
                images, labels, img_path, gt_path, modal = data['image'], data['gt'], data['img_path'], data['gt_path'], data['visible']
                images = torch.cat((images,modal), dim = 1)
            else:
                images, labels, img_path, gt_path = data['image'], data['gt'], data['img_path'], data['gt_path']
            
            
            i += images.size()[0]
            

            #torch.all(labels[0][0].eq(tens_b[0][0]))

            #labels = labels[:,0:1,:,:] #new: mask is read as 3channel image, since 3 channels are equal, take the first one
            #labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()
            images = images.cuda()
            labels = labels.cuda()
            if param_dict['extra_loss']:
                outputs, outputs_f, outputs_b = model(images)  # get prediction
            
            else:
            
              if param_dict['bayesian']:
                    for steps in range(param_dict['by_steps']):              
                        output_list.append(torch.unsqueeze(model(images),0))
                        
                    outputs = torch.cat(output_list, 0).mean(dim=0)
                    unct = torch.cat(output_list, 0).var(dim=0)
                    
         
              else:
                    outputs = model(images)
                    
            
            
            
            vallosses = criterion(outputs, labels)
            pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])
            if param_dict['bayesian']:
              
              unct = tools.utils.out2pred(unct, param_dict['num_class'], param_dict['thread'], unct = True)
            
            batch_num += images.size()[0]
            val_loss += vallosses.item()
            if param_dict['val_visual']:
                for kk in range(len(img_path)):
                    cur_name = os.path.basename(img_path[kk])
                    
                    pred_sub = pred[kk, :, :]
                      
                    label_all[i] = read_image(gt_path[kk], 'gt_val') 
                    
                    
                    predict_all[i] = pred_sub
                    
                    plot_multi(pred_sub, os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice', cur_name))

                    #yimage.io.write_image(
                    #    os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice', cur_name),
                    #    pred_sub)
                        #color_table=param_dict['color_table'])
                    
                    #yimage.io.write_image(
                    #    os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 't_slice', cur_name),
                    #    pred_sub,
                    #    color_table=param_dict['color_table'])
                        
                    if param_dict['bayesian']:
                      
                      
                      unct_sub = unct[kk, :, :]
                      print(cur_name)
                      
                      
                      yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch), 'slice', cur_name),
                        unct_sub,
                        color_table=param_dict['color_table'])
                    
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)))
        
        val_loss = val_loss / batch_num
    return IoU[1], OA, f1ccore[1], val_loss


def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None


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

    if param_dict['amodal'] == True:
        input_bands = param_dict['input_bands']+1
    else:
        input_bands = param_dict['input_bands']

    # Create network structure
    frame_work = get_net(param_dict['model_name'], input_bands, param_dict['num_class'],
                         param_dict['img_size'], param_dict['pretrained_model'])
    
    

    if param_dict['vis_graph']:
        sampledata = torch.rand((1, input_bands, param_dict['img_size'], param_dict['img_size']))
        o = frame_work(sampledata)
        onnx_path = os.path.join(param_dict['save_dir_model'], "model_vis.onnx")
        torch.onnx.export(frame_work, sampledata, onnx_path, opset_version=11)
        netron.start(onnx_path)

    if os.path.exists(param_dict['model_dir']) is False:
        print('Create dir')
        os.mkdir(param_dict['model_dir'])
    
    main()
