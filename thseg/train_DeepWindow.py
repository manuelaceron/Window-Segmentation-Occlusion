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
from tools.losses import get_loss
from tools.parse_config_yaml import parse_yaml
import torch.onnx
import pdb
import matplotlib.pyplot as plt

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_iou = np.inf

    def early_stop(self, validation_iou):
        if validation_iou > self.min_validation_iou:
            self.min_validation_iou = validation_iou
            self.counter = 0
        elif validation_iou < (self.min_validation_iou + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def main():
    early_stopper = EarlyStopper(patience=20, min_delta=0.01)
    composed_transforms_train = standard_transforms.Compose([
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

  
    train_dataset = AmodalSegmentation(txt_path=param_dict['train_list'], transform=composed_transforms_train)  # get data
    val_dataset = AmodalSegmentation(txt_path=param_dict['val_list'], transform=composed_transforms_val)  # get data 
    
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
    optimizer = create_optimizer_v2(model, 'adam', lr=param_dict['base_lr'])#, weight_decay=param_dict['weight_decay']) 
    

    #optimizer = optim.SGD(model.parameters(), lr=param_dict['base_lr'], momentum=param_dict['momentum'], weight_decay=param_dict['weight_decay'])
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)#50 0.8) 20 0.1

    if param_dict['resume_ckpt']:
        resume_ckpt = param_dict['resume_ckpt']  # check point path
        checkpoint = torch.load(resume_ckpt)  # load checkpoint
        #model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(checkpoint['net'])  # load parameters
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch']  # set initial epoch
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])  # loadlr_scheduler
        print('load the model %s' % param_dict['resume_ckpt'])#param_dict['resume_ckpt'])#find_new_file(param_dict['model_dir']))
    model.cuda()

    criterion = get_loss(param_dict['loss_type'])  # define loss
    L1_loss = nn.L1Loss().cuda()
    writer = SummaryWriter(os.path.join(param_dict['save_dir_model'], 'runs'))

    best_val_acc = 0.0
    with open(os.path.join(param_dict['save_dir_model'], 'log.txt'), 'w') as ff, open(os.path.join(param_dict['save_dir_model'], 'occ_log.txt'), 'w') as cc, open(os.path.join(param_dict['save_dir_model'], 'coarse_log.txt'), 'w') as co:
        start_time = time.time()
        for epoch in range(start_epoch, param_dict['epoches']):
            
            model.train()
            running_loss = 0.0
            batch_num = 0

            epoch_start_time = time.time()
            for i, data in tqdm.tqdm(enumerate(trainloader)):  # get data
                output_list = []
                

                images, labels, occ_mask  = data['image'], data['gt'], data['visible'] #First batch of N images 3x5125x512
                

                path = data['img_path']
                
                i += images.size()[0] #Samples in the batch
                
                
                #labels = labels[:,0:1,:,:] #new: mask is read as 3channel image, since 3 channels are equal, take the first one
                labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()
                occ_mask = occ_mask.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()

                images = images.cuda()
                labels = labels.cuda()
                occ_mask = occ_mask.cuda()

                optimizer.zero_grad()

                
                
                
                if param_dict['bayesian']:
                    for steps in range(param_dict['by_steps']):
                        
                        output_list.append(torch.unsqueeze(model(images),0))
                    outputs = torch.cat(output_list, 0).mean(dim=0)
                    unct = torch.cat(output_list, 0).var(dim=0)
                    #pdb.set_trace()
         
                else:
                    outputs1, outputs2, outputs3, vis_patt = model(images)
                    
                    
                #pdb.set_trace()
                losses1 = criterion(outputs1, occ_mask) #Occlusion mask labels
                losses2 = criterion(outputs2, labels) #coarse Window segmentation labels
                losses3 = criterion(outputs3, labels) #Window segmentation labels

                losses = losses1 + losses2 + losses3
                
                losses.backward()  
                optimizer.step()
                running_loss += losses
                batch_num += images.size()[0]
            
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            print('epoch is {}, train loss is {}, Per epoch time {}'.format(epoch, running_loss.item() / batch_num, per_epoch_ptime ))
            print('Loss occlusion is {}, Loss segmentation is {}'.format(losses1.item() / batch_num, losses2.item() / batch_num ))
            cur_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', cur_lr, epoch)
            writer.add_scalar('train_loss', running_loss / batch_num, epoch)
            lr_schedule.step()
            if epoch % param_dict['save_iter'] == 0:
                
                results = eval(valloader, model, criterion, epoch)
                
                val_miou = results[0]
                val_acc = results[1]
                val_f1 = results[2]
                val_loss = results[3]
                val_miou_mask = results[4]
                val_acc_mask = results[5]
                val_f1_mask = results[6]
                val_loss_mask = results[7]
                val_miou_coarse = results[8]
                val_acc_coarse = results[9]
                val_f1_coarse = results[10]
                val_loss_coarse = results[11]

                writer.add_scalar('val_miou', val_miou, epoch)
                writer.add_scalar('val_acc', val_acc, epoch)
                writer.add_scalar('val_f1', val_f1, epoch)
                writer.add_scalar('val_loss', val_loss, epoch)

                cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(running_loss.item() / batch_num), str(val_loss), str(val_f1),
                    str(val_acc),
                    str(val_miou)
                )


                cc_cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(losses1.item() / batch_num), str(val_loss_mask), str(val_f1_mask),
                    str(val_acc_mask),
                    str(val_miou_mask)
                )

                coarse_cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(losses2.item() / batch_num), str(val_loss_coarse), str(val_f1_coarse),
                    str(val_acc_coarse),
                    str(val_miou_coarse)
                )
                
                print('Occ: ',cc_cur_log)
                print('Coarse Seg: ', coarse_cur_log)
                print('Final Seg: ', cur_log)
                
                ff.writelines(str(cur_log))
                cc.writelines(str(cc_cur_log))
                co.writelines(str(coarse_cur_log))
                
                if epoch >= 10:#val_miou > best_val_acc:
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_schedule': lr_schedule.state_dict(),
                        "epoch": epoch
                    }
                    if val_miou > best_val_acc:
                        name= str(epoch)+'valiou_best.pth'
                    else:
                        name = str(epoch)+'model.pth'
                    torch.save(checkpoint, os.path.join(param_dict['model_dir'], name))
                    best_val_acc = val_miou
                    #if early_stopper.early_stop(best_val_acc):
                    #    print('Early stop break')             
                    #    break

        end_time = time.time()
        total_ptime = end_time - start_time
        print('Total time: ',total_ptime) 

def eval(valloader, model, criterion, epoch):
    
    val_num = valloader.dataset.num_sample
    
    label_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    
    label_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    label_all_coarse = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all_coarse = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    model.eval()
    if param_dict['val_visual']:
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual')) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual'))
        
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch))) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)))
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice'))

        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch))) is False:
            os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch)))
            os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch), 'slice'))

        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', 'coarse', str(epoch))) is False:
            os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'coarse', str(epoch)))
            os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'coarse', str(epoch), 'slice'))

        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', 'vis_patt', str(epoch))) is False:
            os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'vis_patt', str(epoch)))
            os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'vis_patt', str(epoch), 'slice'))

        
    
    
            
    with torch.no_grad():
        
        batch_num = 0
        val_loss = 0.0
        val_loss_mask = 0.0
        val_loss_coarse = 0.0
        for i, data in tqdm.tqdm(enumerate(valloader), ascii=True, desc="validate step"):  # get data
            output_list = []
            
            images, labels, img_path, gt_path, gt_path_mask, occ_mask = data['image'], data['gt'], data['img_path'], data['gt_path'], data['modal_path'], data['visible']          
            
            
            i += images.size()[0]
            

            #torch.all(labels[0][0].eq(tens_b[0][0]))

            #labels = labels[:,0:1,:,:] #new: mask is read as 3channel image, since 3 channels are equal, take the first one
            labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()
            occ_mask = occ_mask.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()

            images = images.cuda()
            labels = labels.cuda()
            occ_mask = occ_mask.cuda()

            if param_dict['extra_loss']:
                outputs, outputs_f, outputs_b = model(images)  # get prediction
            
            else:
            
              if param_dict['bayesian']:
                    for steps in range(param_dict['by_steps']):              
                        output_list.append(torch.unsqueeze(model(images),0))
                        
                    outputs = torch.cat(output_list, 0).mean(dim=0)
                    unct = torch.cat(output_list, 0).var(dim=0)
                    
         
              else:
                    out_occ, coarse_outputs, outputs, vis_patt  = model(images)
                    
            
                      
            vallosses = criterion(outputs, labels)
            pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])

            vallosses_coarse = criterion(coarse_outputs, labels)
            pred_coarse = tools.utils.out2pred(coarse_outputs, param_dict['num_class'], param_dict['thread'])

            vallosses_mask = criterion(out_occ, occ_mask)
            pred_mask = tools.utils.out2pred(out_occ, param_dict['num_class'], param_dict['thread'])

            pred_vis = tools.utils.out2pred(vis_patt, param_dict['num_class'], param_dict['thread'])


            if param_dict['bayesian']:
              
              unct = tools.utils.out2pred(unct, param_dict['num_class'], param_dict['thread'], unct = True)
            
            batch_num += images.size()[0]
            val_loss += vallosses.item()
            val_loss_mask += vallosses_mask.item()
            val_loss_coarse += vallosses_coarse.item()

            if param_dict['val_visual']:
                for kk in range(len(img_path)):
                    cur_name = os.path.basename(img_path[kk])
                    pred_sub = pred[kk, :, :]
                    label_all[i] = read_image(gt_path[kk], 'gt') 
                    predict_all[i] = pred_sub
                    
                    pred_sub_mask = pred_mask[kk, :, :]
                    label_all_mask[i] = read_image(gt_path_mask[kk], 'gt') 
                    predict_all_mask[i] = pred_sub_mask

                    pred_sub_coarse = pred_coarse[kk, :, :]
                    label_all_coarse[i] = read_image(gt_path[kk], 'gt') 
                    predict_all_coarse[i] = pred_sub_coarse

                    pred_sub_vis = pred_vis[kk, :, :]

                    yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice', cur_name),
                        pred_sub,
                        color_table=param_dict['color_table'])

                    yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch), 'slice', cur_name),
                        pred_sub_mask,
                        color_table=param_dict['color_table'])
                    
                    yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', 'coarse', str(epoch), 'slice', cur_name),
                        pred_sub_coarse,
                        color_table=param_dict['color_table']) 

                    yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', 'vis_patt', str(epoch), 'slice', cur_name),
                        pred_sub_vis,
                        color_table=param_dict['color_table'])
                        
                    
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)))

        precision_mask, recall_mask, f1ccore_mask, OA_mask, IoU_mask, mIOU_mask = get_acc_v2(
            label_all_mask, predict_all_mask,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)), 'occlusion_mask.txt')
        
        precision_coarse, recall_coarse, f1ccore_coarse, OA_coarse, IoU_coarse, mIOU_coarse = get_acc_v2(
            label_all_coarse, predict_all_coarse,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)), 'coarse.txt')
        
        val_loss = val_loss / batch_num
        val_loss_mask = val_loss_mask / batch_num
        val_loss_coarse = val_loss_coarse / batch_num

    return IoU[1], OA, f1ccore[1], val_loss, IoU_mask[1], OA_mask, f1ccore_mask[1], val_loss_mask, IoU_coarse[1], OA_coarse, f1ccore_coarse[1], val_loss_coarse


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
