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
from networks.discriminator import NLayerDiscriminator
from tools.losses import get_loss
from tools.newLosses import AdversarialLoss
from tools.parse_config_yaml import parse_yaml
import torch.onnx
import pdb
import matplotlib.pyplot as plt
import pdb
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# weight_init
def weight_init(model, mean, std):
    for m in model._modules:
        normal_init(model._modules[m], mean, std)
    
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def main():

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

    if param_dict['amodal'] == False:
        train_dataset = IsprsSegmentation(txt_path=param_dict['train_list'], transform=composed_transforms_train)  # get data        
        val_dataset = IsprsSegmentation(txt_path=param_dict['val_list'], transform=composed_transforms_val)  # get data TODO: the document is not updated with dataset...
        
    else:        
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
    
    L1_lambda = 10

    # network
    
    netD = NLayerDiscriminator(input_nc=1)
    #weight_init(model, mean=0.0, std=0.02)
    #weight_init(netD, mean=0.0, std=0.02)
    model.cuda()
    netD.cuda()
    

    # loss
    BCE_loss = AdversarialLoss('nsgan').cuda()
    L1_loss = nn.L1Loss().cuda()
    criterion = get_loss(param_dict['loss_type'])  # define loss BCE

    #optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5,0.999))
    #optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5,0.999))

    optimizer = create_optimizer_v2(model, 'adam', lr=param_dict['base_lr'])
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    optimizerD = create_optimizer_v2(netD, 'adam', lr=0.0002)
    lr_scheduleD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=50, gamma=0.8)

    if param_dict['resume_ckpt']:
        resume_ckpt = param_dict['resume_ckpt']  # check point path
        checkpoint = torch.load(resume_ckpt)  # load checkpoint
        model.load_state_dict(checkpoint['net'])  # load parameters
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch']  # set initial epoch
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])  # loadlr_scheduler
        print('load the model %s' % find_new_file(param_dict['model_dir']))
    
    #model.cuda()

    
    writer = SummaryWriter(os.path.join(param_dict['save_dir_model'], 'runs'))

    

    best_val_acc = 0.0
    with open(os.path.join(param_dict['save_dir_model'], 'log.txt'), 'w') as ff:
        start_time = time.time()
        for epoch in range(start_epoch, param_dict['epoches']):

            model.train()
            netD.train()
            
            
            running_loss = 0.0
            
            batch_num = 0
            epoch_start_time = time.time()
            for i, data in tqdm.tqdm(enumerate(trainloader)):  # get data
                

                if param_dict['amodal'] == True: 
                    images, labels, visible_mask = data['image'], data['gt'], data['visible'] #First batch of N images 3x5125x512
                    images = torch.cat((images,visible_mask), dim = 1)
                else: 
                    images, labels  = data['image'], data['gt'] #First batch of N images 3x5125x512

                path = data['img_path']
                
                i += images.size()[0] #Samples in the batch
                
                
                #labels = labels[:,0:1,:,:] #new: mask is read as 3channel image, since 3 channels are equal, take the first one
                labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()
                images = images.cuda()
                labels = labels.cuda()
                batch_num += images.size()[0]
                
                #netD.zero_grad()
                
                # Discriminator Real Loss 
                
                #dis_input_real = data['gt'].cuda() #Dimension 1 is not there...
                #dis_real = netD(dis_input_real)
                #dis_real_loss = BCE_loss(dis_real, True, True)

                # Discriminator Fake Loss
                outputs = model(images)
                #dis_input_fake = outputs.detach()
                #dis_fake = netD(dis_input_fake)
                #dis_fake_loss = BCE_loss(dis_fake, False, True)

                #dis_loss = (dis_real_loss + dis_fake_loss) / 2

                # Discriminator Update               

                # Train Generator
                #gen_loss = 0
                
                #gen_input_fake = outputs
                #gen_fake = netD(gen_input_fake)
                #gen_gan_loss = BCE_loss(gen_fake, True, False) #+ L1_lambda * L1_loss(gen_input_fake, dis_input_real) #* adv_loss_weight #TODO: why weighted?
                #gen_loss += gen_gan_loss

                losses = criterion(outputs, labels)  # calculate loss
                #gen_loss += losses
                

                 # update
                optimizer.zero_grad()
                losses.backward() #gen_loss.backward()        
                optimizer.step()
                running_loss += losses
                #optimizerD.zero_grad()
                #dis_loss.backward()
                #optimizerD.step() 
                        

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
             
            #pdb.set_trace()
            #print('epoch is {}, train Gen loss is {}, Dis loss is {}, Per epoch time {}'.format(epoch, gen_loss.item() / batch_num, dis_loss.item() / batch_num, per_epoch_ptime))
            print('epoch is {}, train loss is {}, Per epoch time {}'.format(epoch, running_loss.item() / batch_num, per_epoch_ptime))
            cur_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', cur_lr, epoch)
            #writer.add_scalar('train_loss', gen_loss / batch_num, epoch)
            writer.add_scalar('train_loss', running_loss / batch_num, epoch)
            #writer.add_scalar('disc_loss', dis_loss / batch_num, epoch)
            lr_schedule.step()
            if epoch % param_dict['save_iter'] == 0:
                val_miou, val_acc, val_f1, val_loss = eval(valloader, model, criterion, epoch)
                writer.add_scalar('val_miou', val_miou, epoch)
                writer.add_scalar('val_acc', val_acc, epoch)
                writer.add_scalar('val_f1', val_f1, epoch)
                writer.add_scalar('val_loss', val_loss, epoch)
                """ cur_log = 'epoch:{}, learning_rate:{}, trainG_loss:{}, trainD_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(gen_loss.item() / batch_num), str(dis_loss.item() / batch_num), str(val_loss), str(val_f1),
                    str(val_acc),
                    str(val_miou)
                ) """
                cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(running_loss.item() / batch_num), str(val_loss), str(val_f1),
                    str(val_acc),
                    str(val_miou)
                )
                print(cur_log)
                ff.writelines(str(cur_log))
                if val_miou > best_val_acc:
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_schedule': lr_schedule.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(checkpoint, os.path.join(param_dict['model_dir'], 'valiou_best.pth'))
                    best_val_acc = val_miou
        
        end_time = time.time()
        total_ptime = end_time - start_time
        print('Total time: ' total_ptime)

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
    with torch.no_grad():
        
        batch_num = 0
        val_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(valloader), ascii=True, desc="validate step"):  # get data
        
            if param_dict['amodal'] == True: 
                images, labels, img_path, gt_path, modal = data['image'], data['gt'], data['img_path'], data['gt_path'], data['visible']
                images = torch.cat((images,modal), dim = 1)
            else:
                images, labels, img_path, gt_path = data['image'], data['gt'], data['img_path'], data['gt_path']
            
            
            i += images.size()[0]
            

            #torch.all(labels[0][0].eq(tens_b[0][0]))

            #labels = labels[:,0:1,:,:] #new: mask is read as 3channel image, since 3 channels are equal, take the first one
            labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long()
            images = images.cuda()
            labels = labels.cuda()
            if param_dict['extra_loss']:
                outputs, outputs_f, outputs_b = model(images)  # get prediction
            else:
                outputs = model(images)
            
            vallosses = criterion(outputs, labels)
            pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])
            batch_num += images.size()[0]
            val_loss += vallosses.item()
            if param_dict['val_visual']:
                for kk in range(len(img_path)):
                    cur_name = os.path.basename(img_path[kk])
                    pred_sub = pred[kk, :, :]
                    label_all[i] = read_image(gt_path[kk], 'gt') 
                    predict_all[i] = pred_sub
                    
                    yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice', cur_name),
                        pred_sub,
                        color_table=param_dict['color_table'])
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)))
        
        val_loss = val_loss / batch_num
    return IoU[1], OA, f1ccore[1], val_loss #TODO: Why IoU[1] instead of mIOU?


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

