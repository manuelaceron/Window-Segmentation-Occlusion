from __future__ import division
from tools.utils import read_image
import sys
#import yimage
import numpy as np
import os, time
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
from tools.detrLoss import DetrHungarianMatcher, DetrLoss
from tools.parse_config_yaml import parse_yaml
import torch.onnx
import pdb
import matplotlib.pyplot as plt
import cv2
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, PretrainedConfig, DetrImageProcessor
from transformers import AdamW
import albumentations as aug
from networks.MAT.networks.mat import Generator, Discriminator
from networks.DeepFillV2.networks import Generator as DFV2_Generator
from networks.DeepFillV2.networks import Discriminator as DFV2_Discriminator
import networks.DeepFillV2.losses as  gan_losses
import networks.DeepFillV2.misc as misc
from networks.discriminator import NLayerDiscriminator
from tools.newLosses import AdversarialLoss

np.seterr(divide='ignore', invalid='ignore')



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


def additional_mask():
    img_size= param_dict['img_size']

    bbox = misc.random_bbox(img_size,img_size)
    regular_mask = misc.bbox2mask(img_size,img_size, bbox).cuda()
    irregular_mask = misc.brush_stroke_mask(img_size,img_size).cuda()
    mask = torch.logical_or(irregular_mask, regular_mask).to(torch.float32)
    return mask


def detrLosses(outputs, targets, matcher):
    
    # Retrieve the matching between the outputs of the last layer and the targets
    indices = matcher(outputs, targets)
    
    # Compute the average number of target boxes across all nodes, for normalization purposes
    num_boxes = sum(len(t["class_labels"]) for t in targets)
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    num_boxes = torch.clamp(num_boxes, min=1).item()

    l_labels = loss_labels(outputs, targets, indices, num_boxes, param_dict['num_class'])
    l_boxes = loss_boxes(outputs, targets, indices, num_boxes)
                    
    loss = l_labels + l_boxes
    return loss

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
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data['image'], dim=[0,2,3])
        channels_squared_sum += torch.mean(data['image']**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
processor = DetrImageProcessor(do_resize =False, do_rescale = False)
def collate_fn(batch):
    # Create a preprocessor
    
    ann_folder = "/home/cero_ma/MCV/window_benchmarks/originals/resized/ecp-ref-occ60/train/labels/"
    tmp = []
    n_batch = {}
    for item in batch: 
        tmp.append(item.values())
    
    inputs = list(zip(*tmp))
      
    
    
    #encoding = processor(images=images, annotations=ann_info, masks_path=ann_folder, return_tensors="pt")

    n_batch["image"] = inputs[0]
    n_batch["gt"] = inputs[1]
    n_batch["img_sf"] = inputs[2]
    n_batch["occ"] = inputs[3]
    n_batch["occ_sf"] = inputs[4]
    n_batch["visible_mask"] = inputs[5]
    n_batch["hidden_mask"] = inputs[6]
    n_batch["img_path"] = inputs[7]
    n_batch["gt_path"] = inputs[8]
    n_batch["occ_path"] = inputs[9]
    n_batch["visible_path"] = inputs[10]
    n_batch["hidden_path"] = inputs[11]
    
    n_batch["pixel_values"] = inputs[12]
    n_batch["pro_target"] = inputs[13]
    
    if param_dict['use-fixed-model']: 
        n_batch['df_fimage'] = inputs[14]
        n_batch['df_fooc'] = inputs[15]

    

    #images = inputs[0]
    #segmentation_maps = inputs[1]

    #batch["original_images"] = inputs[2]
    #batch["original_segmentation_maps"] = inputs[3]

    #labels = [item["pro_target"] for item in batch]
    #batch[-1] = labels

    

    return n_batch

weight_dict = {'loss_ce':1, 'loss_bbox':2, 'loss_giou':1}

def main():
    early_stopper = EarlyStopper(patience=param_dict['stop_pat'], min_delta=param_dict['stop_delta'])
    
    # Transformations for Occlusion training with Segoformer
    sf_transform = aug.Compose([
        aug.Flip(p=0.5),
        #aug.RandomCrop(width=128, height=128), does not work for this dataset and task
        aug.HorizontalFlip(p=0.5),
        aug.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        aug.geometric.rotate.Rotate (limit=[-15, 15]) ])
    
    composed_transforms_train_segF = standard_transforms.Compose([ #crops!
        tr.RandomHorizontalFlip(),
        tr.RandomVerticalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(0.9, 1.1)),
        tr.FixedResize(param_dict['img_size'])])  # data pocessing and data augumentation
    

    composed_transforms_train = standard_transforms.Compose([ #crops!
        tr.RandomHorizontalFlip(),
        tr.RandomVerticalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(0.9, 1.1)),
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor(do_not = {'img_sf', 'occ_sf'})])  # data pocessing and data augumentation
    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor(do_not = {'img_sf', 'occ_sf'})])  # data pocessing and data augumentation

    
    if param_dict['two-steps'] or param_dict['three-steps']:
        feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
        
            
        train_dataset = AmodalSegmentation(txt_path=param_dict['train_list'], transform=composed_transforms_train, sf_transform=sf_transform, occSegFormer = True, feature_extractor= feature_extractor, processor=processor)  # get data
        val_dataset = AmodalSegmentation(txt_path=param_dict['val_list'], transform=composed_transforms_val, sf_transform=sf_transform, occSegFormer = True, feature_extractor= feature_extractor, processor=processor)  # get data  
    else:
        feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
        train_dataset = AmodalSegmentation(txt_path=param_dict['train_list'], transform=composed_transforms_train, sf_transform=composed_transforms_train_segF, occSegFormer = True, feature_extractor= feature_extractor, processor=processor)  # get data        
        val_dataset = AmodalSegmentation(txt_path=param_dict['val_list'], transform=composed_transforms_val, sf_transform=sf_transform, occSegFormer = True, feature_extractor= feature_extractor,  processor=processor)  # get data TODO: the document is not updated with dataset...
      
    
    trainloader = DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True,
                                num_workers=param_dict['num_workers'], drop_last=False, collate_fn=collate_fn)  # define traindata new: original was drop_las true

    valloader = DataLoader(val_dataset, batch_size=param_dict['batch_size'], shuffle=False,
                            num_workers=param_dict['num_workers'], drop_last=False, collate_fn=collate_fn)  # define traindata new: original was drop_las true  
                            
    #get_mean_and_std(trainloader)
                                                
    start_epoch = 0
    if not param_dict['adversarial']:
        if len(gpu_list) > 1:
            print('gpu>1')  
            model = torch.nn.DataParallel(frame_work, device_ids=gpu_list)  # use gpu to train
        else:
            model = frame_work
        model.cuda()

        optimizer = create_optimizer_v2(model, 'adam', lr=param_dict['base_lr'])#, weight_decay=param_dict['weight_decay']) 
        #optimizer = AdamW(model.parameters(), lr=param_dict['base_lr'])
    

        #optimizer = optim.SGD(model.parameters(), lr=param_dict['base_lr'], momentum=param_dict['momentum'], weight_decay=param_dict['weight_decay'])
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)#50 0.8) 20 0.1

    if param_dict['fine-tune-DL']:
        print('Fine tuning DL...')
        num_classes = param_dict['num_class']
        checkpoint = torch.load(param_dict['bk_pretrained'], map_location=torch.device('cpu'))
        checkpoint['model_state']['classifier.classifier.3.bias'] = checkpoint['model_state']['classifier.classifier.3.bias'][:num_classes]
        checkpoint['model_state']['classifier.classifier.3.weight'] = checkpoint['model_state']['classifier.classifier.3.weight'][:num_classes]
        model.load_state_dict(checkpoint["model_state"],False)
        print("Model restored from %s" % param_dict['bk_pretrained'])

    

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

        configuration_b2 = PretrainedConfig( 
            architectures = ["SegformerForSemanticSegmentation"],
            attention_probs_dropout_prob = 0.0,
            classifier_dropout_prob = 0.1,
            decoder_hidden_size = 768,  
            depths = [3, 4, 6, 3], 
            downsampling_rates = [1, 4, 8, 16],
            drop_path_rate = 0.1,
            hidden_act = 'gelu',
            hidden_dropout_prob = 0.0,
            hidden_sizes = [64, 128, 320, 512], 
            id2label = { "0": "background", "1": "occlusion"},
            image_size = 224, # I think is not used...
            initializer_range = 0.02,
            label2id = { "background": 0, "occlusion": 1 },
            layer_norm_eps = 1e-06,
            mlp_ratios = [4, 4, 4, 4], 
            model_type = "segformer",
            num_attention_heads = [1, 2, 5, 8], 
            num_channels = 3, 
            num_encoder_blocks = 4, 
            patch_sizes = [7, 3, 3, 3], #[7, 3, 3, 3], # Patch size (same b0-b5)
            reshape_last_stage = True,
            semantic_loss_ignore_index = 255,
            sr_ratios = [8, 4, 2, 1], # Reduction ratio of the efficient Self-Att (same b0-b5)
            strides = [4, 2, 2, 2], #[4, 2, 2, 2], # Stride between 2 adjacent patches (same b0-b5)
            ignore_mismatched_sizes=True,
            torch_dtype = "float32",
            transformers_version = "4.18.0"
            ) 
        
        occ_model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path = pre_model, ignore_mismatched_sizes=True, 
                                                            config = configuration_b2)

        occ_model = torch.nn.DataParallel(occ_model, device_ids=[0])
        occ_model.load_state_dict(state_dict)
        print('epoch: ', torch.load(checkpoint_path)['epoch'])

        occ_model.eval()

        for param in occ_model.parameters():
            param.requires_grad = False
        occ_model.cuda()
        

    
    
    if param_dict['use-fixed-visi-model']:
        checkpoint_path = '/home/cero_ma/MCV/code220419_windows/0401_files/Res_UNet_50_ecp_visible/pth_Res_UNet_50/220valiou_best.pth'  # load checkpoint
        state_dict = torch.load(checkpoint_path)['net']

        visi_model = get_net('Res_UNet_50', 3, 1, 512, None)
        visi_model.load_state_dict(state_dict)
        print('epoch: ', torch.load(checkpoint_path)['epoch'])

        visi_model.eval()

        for param in visi_model.parameters():
            param.requires_grad = False

        
        visi_model.cuda()
    
    if param_dict['adversarial']:
        im_channel = 1
        # MAT
        if param_dict['inp_model'] == "MAT":
            G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=param_dict['img_size'], img_channels=im_channel) #
            #D = Discriminator(c_dim=0, img_resolution=param_dict['img_size'], img_channels=im_channel)
            z = torch.randn(param_dict['batch_size'], 512).cuda()
            D = NLayerDiscriminator(input_nc=im_channel+1)
            
            optimizerG = torch.optim.Adam(G.parameters(), lr=param_dict['base_lr'], betas=[0, 0.99], eps=1e-8) 
            optimizerD = torch.optim.Adam(D.parameters(), lr=param_dict['base_lr'], betas=[0, 0.99], eps=1e-8) #TODO: is working with this???
            lr_schedule = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.8)            
            lr_scheduleD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.8)

        # DeepFillV2
        if param_dict['inp_model'] == "DFV2":
            G = DFV2_Generator(cnum_in=im_channel+2, cnum_out=im_channel, cnum=48, return_flow=False)
            #D = DFV2_Discriminator(cnum_in=im_channel+1, cnum=64)
            D = NLayerDiscriminator(input_nc=im_channel+1)
            gan_loss_d, gan_loss_g = gan_losses.hinge_loss_d, gan_losses.hinge_loss_g

            optimizerG = torch.optim.Adam(G.parameters(), lr=param_dict['base_lr'], betas=[0.5, 0.99]) 
            optimizerD = torch.optim.Adam(D.parameters(), lr=param_dict['base_lr'], betas=[0.5, 0.99])
            lr_schedule = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.8)
            lr_scheduleD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.8)
        

        if len(gpu_list) > 1:
            print('gpu>1')  
            G = torch.nn.DataParallel(G, device_ids=gpu_list)  # use gpu to train
            D = torch.nn.DataParallel(D, device_ids=gpu_list)
        G = G.cuda()
        D = D.cuda()
        
        
        



    # weight occlusions: [0.57487292, 3.83899089]
    # weight visible parts: [0.60757092, 2.8240482 ]
    # hidden windoww: [ 0.5160267  16.09896997]
    # complete windows: [0.63147511 2.40150057]

    occ_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.57487292, 3.83899089])).cuda()   
    vis_criterion = get_loss(param_dict['loss_type_vis'], torch.tensor([0.60757092, 2.8240482])) 
    criterion = get_loss(param_dict['loss_type'], torch.tensor([0.63147511, 2.40150057]))

    #criterionDice = get_loss('DiceLoss')
    #criterionFocal = get_loss('FocalLoss')

    criterionDetr = DetrLoss(
                matcher=DetrHungarianMatcher(class_cost=1, bbox_cost=1, giou_cost=1),
                num_classes=param_dict['num_class'],
                eos_coef=0.1,
            )
    criterionDetr.cuda()
    if param_dict['adversarial']:
        # loss
        adv_loss = AdversarialLoss('nsgan').cuda()
        L1_loss = nn.L1Loss().cuda()
        
    


    #criterion = get_loss(param_dict['loss_type'])  # define loss
    
    L1_loss = nn.L1Loss().cuda()
    writer = SummaryWriter(os.path.join(param_dict['save_dir_model'], 'runs'))

    best_val_acc = 0.0
    aaa = 0
    with open(os.path.join(param_dict['save_dir_model'], 'log.txt'), 'w') as ff, open(os.path.join(param_dict['save_dir_model'], 'occ_log.txt'), 'w') as cc, open(os.path.join(param_dict['save_dir_model'], 'visible_log.txt'), 'w') as vv:
        start_time = time.time()
        #print('Model parameters: ', count_parameters(model))
        
        for epoch in range(start_epoch, param_dict['epoches']):
            
            if param_dict['adversarial']:
                G.train()
                D.train()
                losses = {}
            else:
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
                    images, labels, sf_fimages, gt_occ, sf_focc_labels  = data['image'], data['gt'], data['df_fimage'], data['occ'], data['df_fooc']
                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()
                    gt_occ = gt_occ.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    gt_occ = gt_occ.cuda()

                    
                
                elif param_dict['three-steps']: 
                    images, labels, vis_labels, sf_fimages, gt_occ, sf_focc_labels, hidden_label, hidden_path  = data['image'], data['gt'], data['visible_mask'], data['df_fimage'], data['occ'], data['df_fooc'], data['hidden_mask'], data['hidden_path']

                    vis_labels = vis_labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    vis_labels = vis_labels.cuda()

                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()

                    gt_occ = gt_occ.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    gt_occ = gt_occ.cuda()

                    hidden_label = hidden_label.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    hidden_label = hidden_label.cuda()

                else: 
                
                    #images, labels  = data['image'], data['gt']
                    
                    images = torch.stack(data["image"], dim=0)
                    labels = torch.stack(data["gt"], dim=0)
                    
                    pro_target = data["pro_target"]
                    
                    for v in pro_target:                        
                        v["boxes"] = v["boxes"].cuda() 
                    
                    for v in pro_target:                        
                        v["class_labels"] = v["class_labels"].cuda() 
                    
                
                path = data['img_path']
                
                i += images.size()[0] #Samples in the batch
                
                
                #labels = labels[:,0:1,:,:] #new: mask is read as 3channel image, since 3 channels are equal, take the first one
                
                labels = labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                images = images.cuda()
                labels = labels.cuda()

                if param_dict['adversarial']:
                    optimizerG.zero_grad()
                    optimizerD.zero_grad()
            
                else:
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

                    
                    
                    occ_out = occ_model(sf_fimages, sf_focc_labels, output_attentions=True)
                    upsampled_logits = nn.functional.interpolate(occ_out.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    occ = upsampled_logits.argmax(dim=1) 
                    occ = torch.unsqueeze(occ, 1) 

                    #occ, outputs = model(sf_fimages, sf_focc_labels, images)
                    #images = torch.cat((images,occ), dim=1)
                    #images = images+occ

                    
                    outputs = model(images, occ)
                                                        
                    occ_loss = occ_criterion(upsampled_logits, gt_occ)
                    
                    w_loss = criterion(outputs, labels)

                    #losses = occ_loss + w_loss
                    losses = w_loss
                    
                
                elif param_dict['three-steps']:

                    occ = occ_model(sf_fimages, sf_focc_labels)
                    occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    visible, outputs, ft_maps = model(images,occ)
                                        
                    occ_loss = occ_criterion(occ, gt_occ)
                    w_loss = vis_criterion(visible, vis_labels)
                    complete_loss = criterion(outputs, labels)

                    losses = occ_loss + w_loss + complete_loss


                elif param_dict['adversarial']:
                    # Image to inpaint
                    visible = visi_model(images)
                    #act = nn.Sigmoid()
                    #visible = act(visible)
                    
                    

                    # Mask 
                    sf_fimages = torch.stack(data["df_fimage"], dim=0)
                    sf_focc_labels = torch.stack(data["df_fooc"], dim=0)
                    
                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()
                    
                    occ = occ_model(sf_fimages, sf_focc_labels)
                    occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    occ = occ.argmax(dim=1)  
                    occ = torch.unsqueeze(occ, 1)

                    #TODO: if there is not occlusion, generate random occlusion
                    

                    
                    
                    
                    if param_dict['inp_model'] == 'MAT':
                        #thread = 0.5
                        #visible[visible >= thread] = 1
                        #visible[visible < thread] = 0

                        occ_mask = occ.float()

                        mask_2 = additional_mask()
                        mask = torch.logical_or(occ_mask, mask_2).to(torch.float32)
                    
                        mask = 1 - mask

                        # Discriminator Real Loss 
                        
                        
                        labels = torch.unsqueeze(labels,1).float()
                        
                        dis_real = D(torch.cat([visible, labels],1)) 
                        dis_real_loss = adv_loss(dis_real, True, True)

                        # Discriminator Fake Loss
                    
                        img, img_stg1 = G(visible, mask, z, None, return_stg1=True)
                                            
                        dis_input_fake = img.detach()
                        dis_input_fake_stg1 = img_stg1.detach()

                        dis_fake = D(torch.cat([visible, dis_input_fake],1))
                        dis_fake_stg1 = D(torch.cat([visible, dis_input_fake_stg1],1))

                        dis_fake_loss = adv_loss(dis_fake, False, True)
                        dis_fake_loss_stg1 = adv_loss(dis_fake_stg1, False, True)

                        dis_loss = (dis_real_loss + dis_fake_loss + dis_fake_loss_stg1).mean()

                        # Train Generator
                        gen_loss = 0
                                            
                        gen_input_fake = img
                        gen_input_fake_stg1 = img_stg1
                        gen_fake = D(torch.cat([visible, gen_input_fake],1))
                        gen_fake_stg1 = D(torch.cat([visible, gen_input_fake_stg1],1))
                        # The mean is just a test, all the implementations so far dont use mean...
                        gen_gan_loss = adv_loss(gen_fake, True, False) + adv_loss(gen_fake_stg1, True, False)+ L1_loss(gen_input_fake, labels) + criterion(gen_input_fake, torch.squeeze(labels,1)) + criterion(gen_input_fake_stg1, torch.squeeze(labels,1))
                        gen_loss += gen_gan_loss.mean()

                        # update
                    
                        gen_loss.backward()        
                        optimizerG.step()
                        running_loss += gen_loss
                        
                        dis_loss.backward()
                        optimizerD.step()

                    elif param_dict['inp_model'] == "DFV2":
                        
                        batch_real = visible
                        
                        occ_mask = occ.float()
                        mask_2 = additional_mask()
                        mask = torch.logical_or(occ_mask, mask_2).to(torch.float32)
                        
                        #cv2.imshow('mask_2', (np.asarray(mask_2[0][0].detach().cpu())*255).astype(float))
                        #cv2.imshow('new_mask', (np.asarray(new_mask[0][0].detach().cpu())*255).astype(float))
                           
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        
                        

                        batch_incomplete = batch_real*(1.-mask)
                        ones_x = torch.ones_like(batch_incomplete)[:, 0:1].cuda()
                        x = torch.cat([batch_incomplete, ones_x, ones_x*mask], axis=1)

                        labels = torch.unsqueeze(labels,1).float()
                        
                        dis_real = D(torch.cat([visible, labels],1)) 
                        dis_real_loss = adv_loss(dis_real, True, True)  
                        
                        # Generator coarse and fine stages
                        
                        #x1, x2 = G(x, mask)
                        img_stg1, img = G(x, mask)

                        ##############################

                        dis_input_fake = img.detach()
                        dis_input_fake_stg1 = img_stg1.detach()

                        dis_fake = D(torch.cat([visible, dis_input_fake],1))
                        dis_fake_stg1 = D(torch.cat([visible, dis_input_fake_stg1],1))

                        dis_fake_loss = adv_loss(dis_fake, False, True)
                        dis_fake_loss_stg1 = adv_loss(dis_fake_stg1, False, True)

                        dis_loss = (dis_real_loss + dis_fake_loss + dis_fake_loss_stg1).mean()

                        # Train Generator
                        gen_loss = 0
                                            
                        gen_input_fake = img
                        gen_input_fake_stg1 = img_stg1
                        gen_fake = D(torch.cat([visible, gen_input_fake],1))
                        gen_fake_stg1 = D(torch.cat([visible, gen_input_fake_stg1],1))
                        gen_gan_loss = adv_loss(gen_fake, True, False) + adv_loss(gen_fake_stg1, True, False)+ L1_loss(gen_input_fake, labels) + criterion(gen_input_fake, torch.squeeze(labels,1)) + criterion(gen_input_fake_stg1, torch.squeeze(labels,1))
                        gen_loss += gen_gan_loss 

                        # update
                    
                        gen_loss.backward()        
                        optimizerG.step()
                        running_loss += gen_loss
                        
                        dis_loss.backward()
                        optimizerD.step()

                        ###################################
                        #batch_predicted = x2 



                        # apply mask and complete image
                        #batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)

                        """ if mask[0].max() == 1:
                            cv2.imshow('labels', (np.asarray(labels[0][0].cpu())*255).astype(float))
                            cv2.imshow('visible', (np.asarray(visible[0][0].cpu())*255).astype(float))
                            cv2.imshow('mask', (np.asarray(mask[0][0].cpu())*255).astype(float))
                            cv2.imshow('mask-inv', (np.asarray(1-mask[0][0].cpu())*255).astype(float))
                            cv2.imshow('incomplete', (np.asarray(batch_incomplete[0][0].cpu())*255).astype(float))
                            cv2.imshow('ones_x', (np.asarray(ones_x[0][0].cpu())*255).astype(float))
                            cv2.imshow('ones_x*mask', (np.asarray((ones_x[0][0]*mask[0][0]).cpu())*255).astype(float))
                            cv2.imshow('x2', (np.asarray(x2[0][0].detach().cpu())*255).astype(float))
                            cv2.imshow('batch_complete', (np.asarray(batch_complete[0][0].detach().cpu())*255).astype(float))
                            
                            cv2.waitKey(0)
                            cv2.destroyAllWindows() """ 

                            
                        
                        # D training steps:
                        """ batch_real_mask = torch.cat( (labels, mask), dim=1) # batch_real
                        
                        batch_filled_mask = torch.cat((batch_complete.detach(), mask), dim=1)

                        batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask))

                        d_real_gen = D(batch_real_filled)
                        d_real, d_gen = torch.split(d_real_gen, param_dict['batch_size'])
                        

                        d_loss = gan_loss_d(d_real, d_gen)
                        losses['d_loss'] = d_loss

                        # update D parameters
                        optimizerD.zero_grad()
                        losses['d_loss'].backward()
                        optimizerD.step()

                        # G training steps:
                        losses['ae_loss1'] = 1 * torch.mean((torch.abs(labels - x1))) #criterion(x1, torch.squeeze(labels,1)) 
                        losses['ae_loss2'] = 1 * torch.mean((torch.abs(labels - x2))) #criterion(x2, torch.squeeze(labels,1)) 
                        losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']

                        batch_gen = batch_predicted
                        batch_gen = torch.cat((batch_gen, mask), dim=1)

                        d_gen = D(batch_gen)

                        
                        g_loss = gan_loss_g(d_gen)
                        losses['g_loss'] = g_loss
                        losses['g_loss'] = 1 * losses['g_loss']
                        losses['g_loss'] += losses['ae_loss']

                        # update G parameters
                        optimizerG.zero_grad()
                        losses['g_loss'].backward()
                        optimizerG.step()
                        
                        running_loss += losses['g_loss']
                        
                        
                        
                        aaa += 1 """

                        

                else:
                    
                    
                    #visible = visi_model(images)
                    
                    
                    #images = torch.cat((images,visible), dim=1)
                    sf_fimages = torch.stack(data["df_fimage"], dim=0)
                    sf_focc_labels = torch.stack(data["df_fooc"], dim=0)
                    
                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()
                    
                    occ = occ_model(sf_fimages, sf_focc_labels)
                    occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    occ = occ.argmax(dim=1)  
                    occ = torch.unsqueeze(occ, 1)
                    
                    outputs, add_out = model(images, occ.float())

                    #loss_dict = criterionDetr(add_out, pro_target)
                    
                    #detr_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    
                    mask_losses = criterion(outputs, labels) #+ criterionFocal(outputs, labels) + criterionDice(outputs, labels) 
                    losses = mask_losses #2*mask_losses + detr_losses 
                
                if not param_dict['adversarial']:
                    losses.backward()  
                    optimizer.step()
                    running_loss += losses
                batch_num += images.size()[0]
            
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            
            if param_dict['two-steps']:
                print('Loss occlusion is {}, Loss segmentation is {}'.format(occ_loss.item() / batch_num, w_loss.item() / batch_num ))
            if param_dict['three-steps']:
                print('Loss occlusion is {}, Loss vis segmentation is {}, Loss complete segmentation is {}'.format(occ_loss.item() / batch_num, 
                w_loss.item() / batch_num, complete_loss.item() / batch_num  ))
            print('epoch is {}, train loss is {}, Per epoch time {}'.format(epoch, running_loss.item() / batch_num, per_epoch_ptime ))
            
            if not param_dict['adversarial']:
                cur_lr = optimizer.param_groups[0]['lr']
            else:
                 cur_lr = optimizerG.param_groups[0]['lr']
            writer.add_scalar('learning_rate', cur_lr, epoch)
            writer.add_scalar('train_loss', running_loss / batch_num, epoch)
            
            lr_schedule.step() #TODO: Uncomment this!
            
            if epoch % param_dict['save_iter'] == 0:

                if param_dict['two-steps']:
                    results = eval(valloader, model, occ_model, criterion, occ_criterion, epoch)

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
                    results = eval(valloader, model, occ_model, vis_criterion, criterion, occ_criterion, epoch)

                    val_miou_vis = results[0]
                    val_acc_vis = results[1]
                    val_f1_vis = results[2]
                    val_loss_vis = results[3]

                    val_miou_mask = results[4]
                    val_acc_mask = results[5]
                    val_f1_mask = results[6]
                    val_loss_mask = results[7]

                    val_miou = results[8]
                    val_acc = results[9]
                    val_f1 = results[10]
                    val_loss = results[11]

                    cc_cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(occ_loss.item() / batch_num), str(val_loss_mask), str(val_f1_mask),
                    str(val_acc_mask),
                    str(val_miou_mask)
                    )
                    print('Occ: ',cc_cur_log)
                    cc.writelines(str(cc_cur_log))

                    vis_cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(w_loss.item() / batch_num), str(val_loss_vis), str(val_f1_vis),
                    str(val_acc_vis),
                    str(val_miou_vis)
                    )
                    print('Visible: ',vis_cur_log)
                    vv.writelines(str(vis_cur_log))

                
                
                elif param_dict['adversarial']:
                    val_miou, val_acc, val_f1, val_loss = eval(valloader=valloader, model=G, model2 = occ_model, model3 = visi_model, criterion = criterion, criterion2=adv_loss, criterion3= L1_loss, epoch= epoch)

                else:
                    val_miou, val_acc, val_f1, val_loss = eval(valloader=valloader, model=model, model2 = occ_model, criterion = criterion, criterionDetr=criterionDetr, epoch= epoch)

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
                
                if epoch >= 3:#val_miou > best_val_acc:
                    if param_dict['adversarial']:
                        checkpoint = {
                                "net": G.state_dict(),
                                'optimizer': optimizerG.state_dict(),
                                "epoch": epoch,
                                'lr_schedule': lr_schedule.state_dict()}
                    else: 
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

#ef eval(valloader, model, occ_model, criterion, criterion_2, epoch):
#def eval(valloader, model, model2, criterion, criterionDetr, epoch):
#def eval(valloader, model, model2, criterion, epoch):
def eval(valloader, model, model2, model3, criterion, criterion2, criterion3, epoch):
    
    val_num = valloader.dataset.num_sample
    label_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    
    if param_dict['two-steps']:
        label_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    if param_dict['three-steps']:
        label_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_mask = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        label_all_vis = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_vis = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

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

            if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', 'visible', str(epoch))) is False:
                os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'visible', str(epoch)))
                os.makedirs(os.path.join(param_dict['save_dir_model'], 'val_visual', 'visible', str(epoch), 'slice'))
    
    
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
        val_loss_vis = 0.0
        n = 0
        for i, data in tqdm.tqdm(enumerate(valloader), ascii=True, desc="validate step"):  # get data
            output_list = []
            
            if param_dict['amodal']: 
                images, labels, img_path, gt_path, modal = data['image'], data['visible_mask'], data['img_path'], data['visible_path'], data['occ']
                #images = torch.cat((images,modal), dim = 1)
                images = images+modal
            
            elif param_dict['two-steps']: 

                    images, labels, sf_fimages, gt_occ, sf_focc_labels  = data['image'], data['gt'], data['df_fimage'], data['occ'], data['df_fooc']

                    img_path = data['img_path'] # RGB image path
                    gt_path  = data['gt_path'] #Visible path
                    gt_path_mask  = data['occ_path'] # Occluder path

                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()
                    gt_occ = gt_occ.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    gt_occ = gt_occ.cuda()

            elif param_dict['three-steps']: 

                    images, labels, gt_path, vis_labels, sf_fimages, gt_occ, hidden_label, hidden_path  = data['image'], data['gt'], data['gt_path'], data['visible_mask'], data['df_fimage'], data['occ'], data['hidden_mask'], data['hidden_path']

                    img_path = data['img_path'] # RGB image path
                    gt_path_vis  = data['visible_path'] # Visible path
                    gt_path_mask  = data['occ_path'] # Occluder path
                    gt_path = data['gt_path'] #GT path

                    sf_fimages = sf_fimages.cuda()

                    vis_labels = vis_labels.cuda()
                    vis_labels = vis_labels.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 

                    hidden_label = hidden_label.cuda()
                    hidden_label = hidden_label.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 

                    gt_occ = gt_occ.cuda()
                    gt_occ = gt_occ.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    
            
            else:
                #loaded_data = data['image'], data['gt'], data['img_path'], data['gt_path'], data['occ_path'], data['occ'], data['visible_mask'], data['hidden_mask'], data['visible_path'], data['hidden_path']

                #images = loaded_data[0] #RGB image
                #labels = loaded_data[1] #Window segmentation GT
                #img_path = loaded_data[2] # RGB image path
                #gt_path  = loaded_data[3] #Visible path
                
                #gt_path  = loaded_data[3] #Window segmentation GT path
                #gt_path_mask  = loaded_data[4] # Occluder path
                #occ_mask = loaded_data[5] # Occluder image
                #visible_mask = loaded_data[6] # Visible windows GT
                #hidden_mask = loaded_data[7] # Hidden windows GT

                images = torch.stack(data["image"], dim=0)
                labels = torch.stack(data["gt"], dim=0)
                
                img_path = np.stack(data["img_path"])
                gt_path = np.stack(data["gt_path"])
                pro_target = data["pro_target"]

                for v in pro_target:                        
                    v["boxes"] = v["boxes"].cuda() 
                   
                for v in pro_target:                        
                    v["class_labels"] = v["class_labels"].cuda() 

            
            
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

                    occ_out = occ_model(sf_fimages, sf_focc_labels, output_attentions=True)
                    upsampled_logits = nn.functional.interpolate(occ_out.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    occ = upsampled_logits.argmax(dim=1) 
                    occ = torch.unsqueeze(occ, 1) 

                    #images = torch.cat((images,out_occ), dim=1)
                    #images = images+out_occ
                    outputs = model(images, occ)

                    #out_occ, outputs = model(sf_fimages, sf_focc_labels, images)
                      
              elif param_dict['three-steps']:
                    occ = occ_model(sf_fimages)
                    occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    visible, outputs, ft_maps = model(images,occ)
                                 
              elif param_dict['adversarial']:

                visible = model3(images)
                
                
                sf_fimages = torch.stack(data["df_fimage"], dim=0)
                sf_focc_labels = torch.stack(data["df_fooc"], dim=0)
                    
                sf_fimages = sf_fimages.cuda()
                sf_focc_labels = sf_focc_labels.cuda()
                    
                occ = model2(sf_fimages, sf_focc_labels)
                occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                occ = occ.argmax(dim=1)  
                occ = torch.unsqueeze(occ, 1)

                if param_dict['inp_model'] == "MAT":
                    #thread = 0.5
                    #visible[visible >= thread] = 1
                    #visible[visible < thread] = 0

                    mask = 1 - occ.float()   
                    c = torch.zeros([param_dict['batch_size'], 0]).cuda() #Did i used initially woth this?
                    z = torch.randn(param_dict['batch_size'], 512).cuda()
                    
                    outputs  = model(visible, mask, z, c) #None ->c
                    mask_losses = criterion(outputs,labels) 
                    
                    vallosses = mask_losses

                elif param_dict['inp_model'] == "DFV2":
                    mask = occ.float()   
                    image_masked = visible * (1.-mask)  # mask image

                    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
                    x = torch.cat([image_masked, ones_x, ones_x*mask],dim=1)  # concatenate channels

                    _, x_stage2 = model(x, mask)

                    # complete image
                    outputs = visible * (1.-mask) + x_stage2 * mask
                    
                    mask_losses = criterion(outputs,labels)
                    vallosses = mask_losses


              else:
                    #visible = model2(images)
                    #images = torch.cat((images,visible), dim=1)

                    sf_fimages = torch.stack(data["df_fimage"], dim=0)
                    sf_focc_labels = torch.stack(data["df_fooc"], dim=0)
                    
                    sf_fimages = sf_fimages.cuda()
                    sf_focc_labels = sf_focc_labels.cuda()
                    
                    occ = model2(sf_fimages, sf_focc_labels)
                    occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                    occ = occ.argmax(dim=1)  
                    occ = torch.unsqueeze(occ, 1)


                    outputs, add_out = model(images, occ.float())

                    #loss_dict = criterionDetr(add_out, pro_target)
                    #detr_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    
                    mask_losses = criterion(outputs, labels) 
                    vallosses = mask_losses #2*mask_losses + detr_losses
                    

            if param_dict['two-steps']:
                
                vallosses_mask = criterion_2(upsampled_logits, gt_occ)
                pred_mask = tools.utils.out2pred(upsampled_logits, param_dict['num_class']+1, param_dict['thread'])
                val_loss_mask += vallosses_mask.item()

            if param_dict['three-steps']:
                
                vallosses_mask = criterion_2(occ, gt_occ)
                pred_mask = tools.utils.out2pred(occ, param_dict['num_class']+1, param_dict['thread'])
                val_loss_mask += vallosses_mask.item()

                vallosses_vis = vis_criterion(visible, vis_labels)
                pred_vis = tools.utils.out2pred(visible, param_dict['num_class'], param_dict['thread'])
                val_loss_vis += vallosses_vis.item()

            #vallosses = criterion(outputs, labels)
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
                    

                    #yimage.io.write_image( 
                    cv2.imwrite(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        (pred_sub*255).astype(float))
                        #color_table=param_dict['color_table'])

                    #################### 2 steps: Occ + window #######################
                    if param_dict['two-steps']:
                        pred_sub_mask = pred_mask[kk, :, :]
                        label_all_mask[n] = read_image(gt_path_mask[kk], 'gt') 
                        predict_all_mask[n] = pred_sub_mask

                        #yimage.io.write_image(
                        cv2.imshow(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        pred_sub_mask,
                        color_table=param_dict['color_table'])
                    
                    if param_dict['three-steps']:
                        pred_sub_mask = pred_mask[kk, :, :]
                        label_all_mask[n] = read_image(gt_path_mask[kk], 'gt') 
                        predict_all_mask[n] = pred_sub_mask

                        #yimage.io.write_image(
                        cv2.imshow(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', 'occ', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        pred_sub_mask,
                        color_table=param_dict['color_table'])

                        pred_sub_vis = pred_vis[kk, :, :]
                        label_all_vis[n] = read_image(gt_path_vis[kk], 'gt') 
                        predict_all_vis[n] = pred_sub_vis

                        #yimage.io.write_image(
                        cv2.imwrite(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', 'visible', str(epoch), 'slice', cur_name.split('.')[0]+'.png'),
                        pred_sub_vis)
                        #color_table=param_dict['color_table'])
                    
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

            print('Visible')
            precision_vis, recall_vis, f1ccore_vis, OA_vis, IoU_vis, mIOU_vis= get_acc_v2(
                label_all_vis, predict_all_vis,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model'], 'val_visual', 'visible', str(epoch)), 'visible.txt')
            val_loss_vis = val_loss_vis / batch_num

    if param_dict['two-steps']:
        return IoU[1], OA, f1ccore[1], val_loss, IoU_mask[1], OA_mask, f1ccore_mask[1], val_loss_mask
    if param_dict['three-steps']:
        return IoU[1], OA, f1ccore[1], val_loss, IoU_mask[1], OA_mask, f1ccore_mask[1], val_loss_mask, IoU_vis[1], OA_vis, f1ccore_vis[1], val_loss_vis
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
    #os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['gpu_id']
    gpu_list = [i for i in range(len(param_dict['gpu_id'].split(',')))]
    gx = torch.cuda.device_count()
    print('useful gpu count is {}'.format(gx))

    if param_dict['amodal'] == True:
        input_bands = param_dict['input_bands']
    else:
        input_bands = param_dict['input_bands']

    if not param_dict['adversarial']:
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
