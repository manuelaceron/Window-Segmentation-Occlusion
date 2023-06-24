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
import cv2
import networkDL as network

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
def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

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

    if param_dict['amodal'] == False:
        train_dataset = AmodalSegmentation(txt_path=param_dict['train_list'], transform=composed_transforms_train)  # get data        
        val_dataset = AmodalSegmentation(txt_path=param_dict['val_list'], transform=composed_transforms_val)  # get data TODO: the document is not updated with dataset...
        
    else:        
        train_dataset = AmodalSegmentation(txt_path=param_dict['train_list'], transform=composed_transforms_train)  # get data
        val_dataset = AmodalSegmentation(txt_path=param_dict['val_list'], transform=composed_transforms_val)  # get data 
    
    trainloader = DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True,
                                num_workers=param_dict['num_workers'], drop_last=False)  # define traindata new: original was drop_las true

    valloader = DataLoader(val_dataset, batch_size=param_dict['batch_size'], shuffle=False,
                            num_workers=param_dict['num_workers'], drop_last=False)  # define traindata new: original was drop_las true                            
    start_epoch = 0
    #if len(gpu_list) > 1:
    #    print('gpu>1')  
    #    model = torch.nn.DataParallel(frame_work, device_ids=gpu_list)  # use gpu to train
    #else:
    #    model = frame_work
    
    # if param_dict['resume_ckpt']:
    #     resume_ckpt = param_dict['resume_ckpt']  # check point path
    #     checkpoint = torch.load(resume_ckpt)  # load checkpoint
    #     #model = torch.nn.DataParallel(model, device_ids=[0])
    #     model.load_state_dict(checkpoint['net'])  # load parameters
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if torch.is_tensor(v):
    #                 state[k] = v.cuda()
    #     start_epoch = checkpoint['epoch']  # set initial epoch
    #     lr_schedule.load_state_dict(checkpoint['lr_schedule'])  # loadlr_scheduler
    #     print('load the model %s' % param_dict['resume_ckpt'])#param_dict['resume_ckpt'])#find_new_file(param_dict['model_dir']))
    
    if param_dict['fine-tune-DL']:
        
        model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=8) #8 or 16???'deeplabv3plus_mobilenet'
        network.convert_to_separable_conv(model.classifier)
        set_bn_momentum(model.backbone, momentum=0.01)

        num_classes = param_dict['num_class']
        last_layer = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) 
        
        model.classifier.classifier[3] = last_layer
        
        if len(gpu_list) > 1:
            print('gpu>1')  
            model = torch.nn.DataParallel(model, device_ids=gpu_list)  # use gpu to train

        # Freeze the base layers
        #for name, param in model.named_parameters():
            
        #    if 'classifier.classifier' not in name:
        #        param.requires_grad = False
        


        #print('Fine tuning DL...')
        #checkpoint = torch.load(param_dict['resume_ckpt'], map_location=torch.device('cpu'))
        #checkpoint['model_state']['classifier.classifier.3.bias'] = checkpoint['model_state']['classifier.classifier.3.bias'][:num_classes]
        #checkpoint['model_state']['classifier.classifier.3.weight'] = checkpoint['model_state']['classifier.classifier.3.weight'][:num_classes]
        #model.load_state_dict(checkpoint["model_state"],False)
       
        
        
        #print("Model restored from %s" % param_dict['resume_ckpt'])
        #del checkpoint  # free memory

        #model = torch.nn.DataParallel(model, device_ids=[0])

    optimizer = create_optimizer_v2(model, 'adam', lr=param_dict['base_lr']) 
    

    #optimizer = optim.SGD(model.parameters(), lr=param_dict['base_lr'], momentum=param_dict['momentum'], weight_decay=param_dict['weight_decay'])
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)#50 0.8) 20 0.1


    model.cuda()
    # weight occlusions: [0.57487292, 3.83899089]
    # weight visible parts: [0.60757092, 2.8240482 ]

    criterion = get_loss(param_dict['loss_type'], torch.tensor([0.60757092, 2.8240482])) # define loss
    L1_loss = nn.L1Loss().cuda()
    writer = SummaryWriter(os.path.join(param_dict['save_dir_model'], 'runs'))

    best_val_acc = 0.0
    with open(os.path.join(param_dict['save_dir_model'], 'log.txt'), 'w') as ff:
        start_time = time.time()
        print('Model parameters: ', count_parameters(model))
        for epoch in range(start_epoch, param_dict['epoches']):
            
            model.train()
            running_loss = 0.0
            batch_num = 0

            epoch_start_time = time.time()
            for i, data in tqdm.tqdm(enumerate(trainloader)):  # get data
                output_list = []
                

                if param_dict['amodal'] == True: 
                    images, labels, modal  = data['image'], data['gt'], data['visible'] #First batch of N images 3x5125x512
                    images = torch.cat((images,modal), dim = 1)
                else: 
                    images, labels  = data['image'], data['visible_mask'] #First batch of N images 3x5125x512

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
                    #pdb.set_trace()
         
                else:
                    outputs = model(images)
                    
                

                losses = criterion(outputs, labels) #+ L1_loss(outputs, data['gt'].cuda()) # calculate loss
                
                losses.backward()  
                optimizer.step()
                running_loss += losses
                batch_num += images.size()[0]
            
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            print('epoch is {}, train loss is {}, Per epoch time {}'.format(epoch, running_loss.item() / batch_num, per_epoch_ptime ))
            cur_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', cur_lr, epoch)
            writer.add_scalar('train_loss', running_loss / batch_num, epoch)
            lr_schedule.step()
            if epoch % param_dict['save_iter'] == 0:
                val_miou, val_acc, val_f1, val_loss = eval(valloader, model, criterion, epoch)
                writer.add_scalar('val_miou', val_miou, epoch)
                writer.add_scalar('val_acc', val_acc, epoch)
                writer.add_scalar('val_f1', val_f1, epoch)
                writer.add_scalar('val_loss', val_loss, epoch)
                cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(running_loss.item() / batch_num), str(val_loss), str(val_f1),
                    str(val_acc),
                    str(val_miou)
                )
                print(cur_log)
                ff.writelines(str(cur_log))
                
                if epoch >= 30:#val_miou > best_val_acc:
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
                    if early_stopper.early_stop(best_val_acc):
                        print('Early stop break')             
                        break

        end_time = time.time()
        total_ptime = end_time - start_time
        print('Total time: ',total_ptime) 

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
        n = 0
        for i, data in tqdm.tqdm(enumerate(valloader), ascii=True, desc="validate step"):  # get data
            output_list = []
            
            if param_dict['amodal'] == True: 
                images, labels, img_path, gt_path, modal = data['image'], data['gt'], data['img_path'], data['gt_path'], data['visible']
                images = torch.cat((images,modal), dim = 1)
            else:
                images, labels, img_path, gt_path = data['image'], data['visible_mask'], data['img_path'], data['visible_path']
            
            
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
                      
                    label_all[n] = read_image(gt_path[kk], 'gt') 
                    
                    
                    predict_all[n] = pred_sub
                    n +=1
                    #plot_multi(pred_sub, os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice', cur_name))

                    yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        pred_sub,
                        color_table=param_dict['color_table'])
                
                        
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
    #frame_work = get_net(param_dict['model_name'], input_bands, param_dict['num_class'],
    #                     param_dict['img_size'], param_dict['pretrained_model'])
    
    

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


