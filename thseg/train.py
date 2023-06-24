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
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers import AdamW


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_iou = -1000

    def early_stop(self, validation_iou):
        if validation_iou > self.min_validation_iou:
            self.min_validation_iou = validation_iou
            self.counter = 0
        elif validation_iou < (self.min_validation_iou + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plot_multi(prediction, path):


    color_map = {
    0: (255, 255, 255),   
    1: (255, 0, 0),   
    2: (0, 255, 0),    
    3: (0, 0, 255),
    4: (255, 128, 0),
    5: (0, 128, 255),
    6: (255, 153, 255)
    }

    # Create an RGB image from the prediction
    rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)

    for i in range(512):
        for j in range(512):
            class_id = prediction[i, j]
            rgb_image[i, j] = color_map[class_id]

    # Save the image
    cv2.imwrite(path, rgb_image)

def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    early_stopper = EarlyStopper(patience=param_dict['stop_pat'], min_delta=param_dict['stop_delta'])
    composed_transforms_train = standard_transforms.Compose([ #crops!
        tr.RandomHorizontalFlip(),
        tr.RandomVerticalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(0.9, 1.1)),
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor()])  # data pocessing and data augumentation
    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor()])  # data pocessing and data augumentation

    
    if param_dict['two-steps'] or param_dict['three-steps']:
        feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
            
        train_dataset = AmodalSegmentation(txt_path=param_dict['train_list'], transform=composed_transforms_train, occSegFormer = True, feature_extractor= feature_extractor)  # get data
        val_dataset = AmodalSegmentation(txt_path=param_dict['val_list'], transform=composed_transforms_val, occSegFormer = True, feature_extractor= feature_extractor)  # get data  
    else:
        train_dataset = AmodalSegmentation(txt_path=param_dict['train_list'], transform=composed_transforms_train)  # get data        
        val_dataset = AmodalSegmentation(txt_path=param_dict['val_list'], transform=composed_transforms_val)  # get data TODO: the document is not updated with dataset...
      
    
    trainloader = DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True,
                                num_workers=param_dict['num_workers'], drop_last=False)  # define traindata new: original was drop_las true

    valloader = DataLoader(val_dataset, batch_size=param_dict['batch_size'], shuffle=False,
                            num_workers=param_dict['num_workers'], drop_last=False)  # define traindata new: original was drop_las true                            
    start_epoch = 0
    
    if len(gpu_list) > 1:
        print('gpu>1')  
        model = torch.nn.DataParallel(frame_work, device_ids=gpu_list)  # use gpu to train
    else:
        model = frame_work

    if param_dict['fine-tune-DL']:
        print('Fine tuning DL...')
        num_classes = param_dict['num_class']
        checkpoint = torch.load(param_dict['bk_pretrained'], map_location=torch.device('cpu'))
        checkpoint['model_state']['classifier.classifier.3.bias'] = checkpoint['model_state']['classifier.classifier.3.bias'][:num_classes]
        checkpoint['model_state']['classifier.classifier.3.weight'] = checkpoint['model_state']['classifier.classifier.3.weight'][:num_classes]
        model.load_state_dict(checkpoint["model_state"],False)
        print("Model restored from %s" % param_dict['bk_pretrained'])

    optimizer = create_optimizer_v2(model, 'adam', lr=param_dict['base_lr'])#, weight_decay=param_dict['weight_decay']) 
    #optimizer = AdamW(model.parameters(), lr=param_dict['base_lr'])
    

    #optimizer = optim.SGD(model.parameters(), lr=param_dict['base_lr'], momentum=param_dict['momentum'], weight_decay=param_dict['weight_decay'])
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)#50 0.8) 20 0.1

    if param_dict['resume_ckpt']:
        resume_ckpt = param_dict['resume_ckpt']  # check point path
        checkpoint = torch.load(resume_ckpt)  # load checkpoint
        #model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(checkpoint['net'])  # load parameters
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch']  # set initial epoch
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])  # loadlr_scheduler
        print('load the model %s' % param_dict['resume_ckpt'])#param_dict['resume_ckpt'])#find_new_file(param_dict['model_dir']))
   
    if param_dict['use-fixed-model']:


        checkpoint_path = '/home/cero_ma/MCV/code220419_windows/0401_files/occ: SegFormer/SegFormer_ecp_occ_b2_loss_aug_lr0_00006_preCityS/250valiou_best.pth'  # load checkpoint
        state_dict = torch.load(checkpoint_path)['net']
        
        #occ_model = get_net('segFormer_b2', 3, 2, param_dict['img_size'], param_dict['pretrained_model'])

        pre_model = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" # "nvidia/mit-b2"
        id2label = {0: 'background', 1: 'occlusion'}
        label2id = {'background': 0, 'occlusion': 1}
        
        occ_model = SegformerForSemanticSegmentation.from_pretrained(pre_model, ignore_mismatched_sizes=True,
                                                            num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                            reshape_last_stage=True)

        occ_model = torch.nn.DataParallel(occ_model, device_ids=[0])
        occ_model.load_state_dict(state_dict)
        print('epoch: ', torch.load(checkpoint_path)['epoch'])

        occ_model.eval()

        for param in occ_model.parameters():
            param.requires_grad = False
        
        occ_model.cuda()
        

    model.cuda()

    # weight occlusions: [0.57487292, 3.83899089]
    # weight visible parts: [0.60757092, 2.8240482 ]
    # hidden windoww: [ 0.5160267  16.09896997]

    w_criterion = get_loss(param_dict['loss_type_w'], torch.tensor([0.60757092, 2.8240482]))  # define loss
    hidden_criterion = get_loss(param_dict['loss_type_w'], torch.tensor([0.5160267, 16.09896997]))
    occ_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.57487292, 3.83899089])).cuda() 


    #criterion = get_loss(param_dict['loss_type'])  # define loss
    
    L1_loss = nn.L1Loss().cuda()
    writer = SummaryWriter(os.path.join(param_dict['save_dir_model'], 'runs'))

    best_val_acc = 0.0
    with open(os.path.join(param_dict['save_dir_model'], 'log.txt'), 'w') as ff, open(os.path.join(param_dict['save_dir_model'], 'occ_log.txt'), 'w') as cc, open(os.path.join(param_dict['save_dir_model'], 'hiden_log.txt'), 'w') as hh:
        start_time = time.time()
        print('Model parameters: ', count_parameters(model))
        
        for epoch in range(start_epoch, param_dict['epoches']):
            
            model.train()
            
            running_loss = 0.0
            batch_num = 0

            epoch_start_time = time.time()
            

            
            for i, data in tqdm.tqdm(enumerate(trainloader)):  # get data

                
                output_list = []
                

                if param_dict['amodal']: 
                    images, labels, modal  = data['image'], data['visible_mask'], data['occ'] #First batch of N images 3x5125x512
                    images = torch.cat((images,modal), dim = 1)
                
                elif param_dict['two-steps']: 
                    images, labels, sf_fimages, gt_occ, sf_focc_labels  = data['image'], data['visible_mask'], data['df_fimage'], data['occ'], data['df_fooc']
                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()
                    gt_occ = gt_occ.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    gt_occ = gt_occ.cuda()
                
                elif param_dict['three-steps']: 
                    images, labels, sf_fimages, gt_occ, sf_focc_labels, hidden_label, hidden_path  = data['image'], data['visible_mask'], data['df_fimage'], data['occ'], data['df_fooc'], data['hidden_mask'], data['hidden_path']

                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()

                    gt_occ = gt_occ.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    gt_occ = gt_occ.cuda()

                    hidden_label = hidden_label.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    hidden_label = hidden_label.cuda()

                else: 
                    images, labels  = data['image'], data['visible_mask'] #data['gt'] #First batch of N images 3x5125x512
                
                path = data['img_path']
                
                i += images.size()[0] #Samples in the batch
                
                
                #labels = labels[:,0:1,:,:] #new: mask is read as 3channel image, since 3 channels are equal, take the first one
                
                labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                images = images.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()

                if param_dict['bayesian']:
                    for steps in range(param_dict['by_steps']):
                        
                        output_list.append(torch.unsqueeze(model(images),0))

                    outputs = torch.cat(output_list, 0).mean(dim=0)
                    unct = torch.cat(output_list, 0).var(dim=0)
                    
                    unct = 1 - unct
                    unct = unct.squeeze(1)
                    weighted_loss = unct * criterion(outputs, labels) #+ L1_loss(outputs, data['gt'].cuda()) # calculate loss 
                    losses = torch.mean(weighted_loss)

                elif param_dict['two-steps']:
                    occ = occ_model(sf_fimages, sf_focc_labels)
                    upsampled_logits = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    occ = upsampled_logits.argmax(dim=1) 
                    occ = torch.unsqueeze(occ, 1) 

                    #occ, outputs = model(sf_fimages, sf_focc_labels, images)
                    #images = torch.cat((images,occ), dim=1)
                    images = images+occ


                    outputs = model(images)
                                                        
                    occ_loss = occ_criterion(upsampled_logits, gt_occ)
                    
                    w_loss = w_criterion(outputs, labels)

                    #losses = occ_loss + w_loss
                    losses = w_loss
                    #pdb.set_trace()
                
                elif param_dict['three-steps']:
                    occ, outputs, hidden = model(sf_fimages, sf_focc_labels, images)
                                        
                    occ_loss = occ_criterion(occ, gt_occ)
                    w_loss = w_criterion(outputs, labels)
                    hidden_loss = hidden_criterion(hidden, hidden_label)

                    losses = occ_loss + w_loss + hidden_loss


                else:
                    outputs = model(images)
                    losses = w_criterion(outputs, labels) #+ L1_loss(outputs, data['gt'].cuda()) # calculate loss 
                
                
                losses.backward()  
                optimizer.step()
                running_loss += losses
                batch_num += images.size()[0]
            
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            
            if param_dict['two-steps']:
                print('Loss occlusion is {}, Loss segmentation is {}'.format(occ_loss.item() / batch_num, w_loss.item() / batch_num ))
            if param_dict['three-steps']:
                print('Loss occlusion is {}, Loss segmentation is {}, Loss hidden segmentation is {}'.format(occ_loss.item() / batch_num, 
                w_loss.item() / batch_num, hidden_loss.item() / batch_num  ))
            print('epoch is {}, train loss is {}, Per epoch time {}'.format(epoch, running_loss.item() / batch_num, per_epoch_ptime ))
            
            cur_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', cur_lr, epoch)
            writer.add_scalar('train_loss', running_loss / batch_num, epoch)
            lr_schedule.step() #TODO: Uncomment this!
            if epoch % param_dict['save_iter'] == 0:

                if param_dict['two-steps']:
                    results = eval(valloader, model, occ_model, w_criterion, occ_criterion, epoch)

                    val_miou = results[0]
                    val_acc = results[1]
                    val_f1 = results[2]
                    val_loss = results[3]
                    val_miou_mask = results[4]
                    val_acc_mask = results[5]
                    val_f1_mask = results[6]
                    val_loss_mask = results[7]

                    cc_cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(occ_loss.item() / batch_num), str(val_loss_mask), str(val_f1_mask),
                    str(val_acc_mask),
                    str(val_miou_mask)
                    )
                    print('Occ: ',cc_cur_log)
                    cc.writelines(str(cc_cur_log))
                
                elif param_dict['three-steps']:
                    results = eval(valloader, model, w_criterion, hidden_criterion, occ_criterion, epoch)

                    val_miou = results[0]
                    val_acc = results[1]
                    val_f1 = results[2]
                    val_loss = results[3]
                    val_miou_mask = results[4]
                    val_acc_mask = results[5]
                    val_f1_mask = results[6]
                    val_loss_mask = results[7]
                    val_miou_hidden = results[8]
                    val_acc_hidden = results[9]
                    val_f1_hidden = results[10]
                    val_loss_hidden = results[11]

                    cc_cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(occ_loss.item() / batch_num), str(val_loss_mask), str(val_f1_mask),
                    str(val_acc_mask),
                    str(val_miou_mask)
                    )
                    print('Occ: ',cc_cur_log)
                    cc.writelines(str(cc_cur_log))

                    hidden_cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(hidden_loss.item() / batch_num), str(val_loss_hidden), str(val_f1_hidden),
                    str(val_acc_hidden),
                    str(val_miou_hidden)
                    )
                    print('Hidden: ',hidden_cur_log)
                    hh.writelines(str(hidden_cur_log))

                
                

                else:
                    val_miou, val_acc, val_f1, val_loss = eval(valloader=valloader, model=model, criterion_w = w_criterion, epoch= epoch)

                writer.add_scalar('val_miou', val_miou, epoch)
                writer.add_scalar('val_acc', val_acc, epoch)
                writer.add_scalar('val_f1', val_f1, epoch)
                writer.add_scalar('val_loss', val_loss, epoch)

                cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(running_loss.item() / batch_num), str(val_loss), str(val_f1),
                    str(val_acc),
                    str(val_miou)
                )
                print('Final Seg: ', cur_log)
                ff.writelines(str(cur_log))
                
                if epoch >= 30:#val_miou > best_val_acc:
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        
                        "epoch": epoch,
                        'lr_schedule': lr_schedule.state_dict()}
                    if val_miou > best_val_acc:
                        name= str(epoch)+'valiou_best.pth'
                    else:
                        name = str(epoch)+'model.pth'
                    torch.save(checkpoint, os.path.join(param_dict['model_dir'], name))
                    best_val_acc = val_miou
                    if early_stopper.early_stop(best_val_acc):
                        print('Early stop break')             
                        break

        end_time = time.time()
        total_ptime = end_time - start_time
        print('Total time: ',total_ptime) 

#def eval(valloader, model, occ_model, criterion_w, criterion_occ, epoch):
def eval(valloader, model, criterion_w, epoch):
    
    val_num = valloader.dataset.num_sample
    label_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    
    if param_dict['two-steps']:
        label_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    if param_dict['three-steps']:
        label_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        label_all_hidden = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_hidden = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    model.eval()
    if param_dict['val_visual']:
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual')) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual'))
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch))) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)))
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice'))
        
        if param_dict['two-steps']:
            if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch))) is False:
                os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch)))
                os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch), 'slice'))
        
        if param_dict['three-steps']:
            if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch))) is False:
                os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch)))
                os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch), 'slice'))

            if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', 'hidden', str(epoch))) is False:
                os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'hidden', str(epoch)))
                os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'hidden', str(epoch), 'slice'))
    
    
    if param_dict['bayesian']:
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'unct_val_visual')) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'unct_val_visual'))
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch))) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch)))
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch), 'slice'))
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch), 't_slice'))
            
    with torch.no_grad():
        
        batch_num = 0
        val_loss = 0.0
        val_loss_mask = 0.0
        val_loss_hidden = 0.0
        n = 0
        for i, data in tqdm.tqdm(enumerate(valloader), ascii=True, desc="validate step"):  # get data
            output_list = []
            
            if param_dict['amodal']: 
                images, labels, img_path, gt_path, modal = data['image'], data['visible_mask'], data['img_path'], data['visible_path'], data['occ']
                #images = torch.cat((images,modal), dim = 1)
                images = images+modal
            
            elif param_dict['two-steps']: 

                    images, labels, sf_fimages, gt_occ, sf_focc_labels  = data['image'], data['visible_mask'], data['df_fimage'], data['occ'], data['df_fooc']

                    img_path = data['img_path'] # RGB image path
                    gt_path  = data['visible_path'] #Visible path
                    gt_path_mask  = data['occ_path'] # Occluder path

                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()
                    gt_occ = gt_occ.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    gt_occ = gt_occ.cuda()

            elif param_dict['three-steps']: 

                    images, labels, sf_fimages, gt_occ, sf_focc_labels, hidden_label, hidden_path  = data['image'], data['visible_mask'], data['df_fimage'], data['occ'], data['df_fooc'], data['hidden_mask'], data['hidden_path']

                    img_path = data['img_path'] # RGB image path
                    gt_path  = data['visible_path'] #Visible path
                    gt_path_mask  = data['occ_path'] # Occluder path
                    hidden_label = hidden_label.cuda()
                    hidden_label = hidden_label.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 

                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()
                    gt_occ = gt_occ.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    gt_occ = gt_occ.cuda()
            
            else:
                loaded_data = data['image'], data['gt'], data['img_path'], data['gt_path'], data['occ_path'], data['occ'], data['visible_mask'], data['hidden_mask'], data['visible_path'], data['hidden_path']

                images = loaded_data[0] #RGB image
                labels = loaded_data[6] #Window segmentation GT
                img_path = loaded_data[2] # RGB image path
                gt_path  = loaded_data[8] #Visible path
                
                #gt_path  = loaded_data[3] #Window segmentation GT path
                #gt_path_mask  = loaded_data[4] # Occluder path
                #occ_mask = loaded_data[5] # Occluder image
                #visible_mask = loaded_data[6] # Visible windows GT
                #hidden_mask = loaded_data[7] # Hidden windows GT

            
            
            i += images.size()[0]
            

            #torch.all(labels[0][0].eq(tens_b[0][0]))

            #labels = labels[:,0:1,:,:] #new: mask is read as 3channel image, since 3 channels are equal, take the first one
            labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()
            images = images.cuda()
            labels = labels.cuda()
            if param_dict['extra_loss']:
                outputs, outputs_f, outputs_b = model(images)  # get prediction
            
            else:
            
              if param_dict['bayesian']:
                    for module in model.modules():
                        
                        if isinstance(module, nn.Dropout):
                            #print(module)
                            module.train()
                    
                    for steps in range(param_dict['by_steps']):              
                        output_list.append(torch.unsqueeze(model(images),0))
                        
                    outputs = torch.cat(output_list, 0).mean(dim=0)
                    unct = torch.cat(output_list, 0).var(dim=0)                    
                    
                    unct = 1 - unct
              
              elif param_dict['two-steps']:

                    occ = occ_model(sf_fimages, sf_focc_labels)
                    upsampled_logits = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    occ = upsampled_logits.argmax(dim=1) 
                    out_occ = torch.unsqueeze(occ, 1) 

                    #images = torch.cat((images,out_occ), dim=1)
                    images = images+out_occ
                    outputs = model(images)

                    #out_occ, outputs = model(sf_fimages, sf_focc_labels, images)
                      
              elif param_dict['three-steps']:
                    out_occ, outputs, hidden = model(sf_fimages, sf_focc_labels, images)
                        
         
              else:
                    outputs = model(images)
                    

            if param_dict['two-steps']:
                
                vallosses_mask = criterion_occ(upsampled_logits, gt_occ)
                pred_mask = tools.utils.out2pred(upsampled_logits, param_dict['num_class']+1, param_dict['thread'])
                val_loss_mask += vallosses_mask.item()

            if param_dict['three-steps']:
                
                vallosses_mask = criterion_occ(out_occ, gt_occ)
                pred_mask = tools.utils.out2pred(out_occ, param_dict['num_class']+1, param_dict['thread'])
                val_loss_mask += vallosses_mask.item()

                vallosses_hidden = hidden_criterion(hidden, hidden_label)
                pred_hidden = tools.utils.out2pred(hidden, param_dict['num_class'], param_dict['thread'])
                val_loss_hidden += vallosses_hidden.item()

            vallosses = criterion_w(outputs, labels)
            pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])
            val_loss += vallosses.item()

            if param_dict['bayesian']:
            
              unc_pred = unct
              unct_t = tools.utils.out2pred(unc_pred, param_dict['num_class'], 0.95, unct = True, thres= True)
              unct = tools.utils.out2pred(unct, param_dict['num_class'], param_dict['thread'], unct = True)
              
            
            batch_num += images.size()[0]
            
            if param_dict['val_visual']:
                for kk in range(len(img_path)):
                    cur_name = os.path.basename(img_path[kk])
                    
                    pred_sub = pred[kk, :, :]                      
                    label_all[n] = read_image(gt_path[kk], 'gt') 
                    predict_all[n] = pred_sub

                    yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        pred_sub,
                        color_table=param_dict['color_table'])

                    #################### 2 steps: Occ + window #######################
                    if param_dict['two-steps']:
                        pred_sub_mask = pred_mask[kk, :, :]
                        label_all_mask[n] = read_image(gt_path_mask[kk], 'gt') 
                        predict_all_mask[n] = pred_sub_mask

                        yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        pred_sub_mask,
                        color_table=param_dict['color_table'])
                    
                    if param_dict['three-steps']:
                        pred_sub_mask = pred_mask[kk, :, :]
                        label_all_mask[n] = read_image(gt_path_mask[kk], 'gt') 
                        predict_all_mask[n] = pred_sub_mask

                        yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        pred_sub_mask,
                        color_table=param_dict['color_table'])

                        pred_sub_hidden = pred_hidden[kk, :, :]
                        label_all_hidden[n] = read_image(hidden_path[kk], 'gt') 
                        predict_all_hidden[n] = pred_sub_hidden

                        yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', 'hidden', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        pred_sub_hidden,
                        color_table=param_dict['color_table'])
                    
                    #################### Bayesian Network #######################   
                    if param_dict['bayesian']:
                        unct_sub = unct[kk, :, :]
                        unct_t_sub = unct_t[kk, :, :]
                        
                        #cv2.imwrite(os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch), 'slice', cur_name), unct_sub*)
                        yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        unct_sub)#,
                        #color_table= [(0, 0, 0), (255, 255, 255)] ) #param_dict['color_table'])
                    
                        yimage.io.write_image(
                            os.path.join(param_dict['save_dir_model'], 'unct_val_visual', str(epoch), 't_slice', cur_name.split('.')[0]+'.png'),
                            unct_t_sub,
                            color_table=param_dict['color_table'])
                    n += 1
        
        print('W. Segmentation')            
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)))
        val_loss = val_loss / batch_num

        if param_dict['two-steps']:
            print('Occlusion')
            precision_mask, recall_mask, f1ccore_mask, OA_mask, IoU_mask, mIOU_mask = get_acc_v2(
                label_all_mask, predict_all_mask,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch)), 'occlusion_mask.txt')
            val_loss_mask = val_loss_mask / batch_num

        if param_dict['three-steps']:
            print('Occlusion')
            precision_mask, recall_mask, f1ccore_mask, OA_mask, IoU_mask, mIOU_mask = get_acc_v2(
                label_all_mask, predict_all_mask,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch)), 'occlusion_mask.txt')
            val_loss_mask = val_loss_mask / batch_num

            print('Hidden')
            precision_hidden, recall_hidden, f1ccore_hidden, OA_hidden, IoU_hidden, mIOU_hidden = get_acc_v2(
                label_all_hidden, predict_all_hidden,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model'], 'val_visual', 'hidden', str(epoch)), 'hidden.txt')
            val_loss_hidden = val_loss_hidden / batch_num

    if param_dict['two-steps']:
        return IoU[1], OA, f1ccore[1], val_loss, IoU_mask[1], OA_mask, f1ccore_mask[1], val_loss_mask
    if param_dict['three-steps']:
        return IoU[1], OA, f1ccore[1], val_loss, IoU_mask[1], OA_mask, f1ccore_mask[1], val_loss_mask, IoU_hidden[1], OA_hidden, f1ccore_hidden[1], val_loss_hidden
    else:
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
        input_bands = param_dict['input_bands']
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
