from __future__ import division
import sys
import os, math
import torchvision
from tools.utils import read_image
from tools.metrics import get_acc_v2
import numpy as np
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader, Dataset
import tqdm
from collections import OrderedDict
import tools.transform as tr
from tools.dataloader import IsprsSegmentation, AmodalSegmentation
import tools
import torch
from networks.get_model import get_net
from tools.parse_config_yaml import parse_yaml
import torch.onnx
import pdb, cv2
import networkDL as network
import torch.nn as nn
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, DetrImageProcessor
from transformers.image_transforms import center_to_corners_format
import albumentations as aug
from timm.utils import AverageMeter
from timm.optim import create_optimizer_v2
from sklearn.metrics import jaccard_score
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from networks.MAT.networks.mat import Generator

def pos_process_boxes(add_out):

    tg= torch.tensor([param_dict['img_size'],param_dict['img_size']])
    tg = tg.repeat(param_dict['batch_size'],1)


    out_logits, out_bbox = add_out['logits'], add_out['pred_boxes']

    prob = nn.functional.softmax(out_logits, -1)
    
    scores, labels = prob[..., :].max(-1)

    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(out_bbox) 
    
    img_h, img_w = tg.unbind(1)
    
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    
    boxes = boxes * scale_fct#[:, None, :]
    #threshold=0.5
    #for s, l, b in zip(scores, labels, boxes):
    #        score = s[s > threshold]
    #        label = l[s > threshold]
    #        box = b[s > threshold]
    #        results.append({"scores": score, "labels": label, "boxes": box})
    pdb.set_trace()
    return boxes

def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()

    return grad_map

def visualize_attention(all_attentions, k, name):
    
    attention_weights = all_attentions[k]
    L = attention_weights.shape[0]
    num_tokens = attention_weights.shape[1]
    
    height = int(math.sqrt(num_tokens))
    width = int(math.sqrt(num_tokens))

    # Normalize attention weights across the source sequence length
    attention_weights = nn.functional.softmax(attention_weights, dim=-1)

    # Iterate over each sample in the batch
    
    # Iterate over each target query (L) and its corresponding attention weights
    for j in range(L):
        # Get the attention weights for the j-th target query
        query_attention_weights = attention_weights[j]
            
        # Resize attention weights to match the spatial dimensions of the input images
        query_attention_weights = query_attention_weights.view(height, width)
            
        # Visualize the attention weights as a heatmap
        imgplot =  plt.imshow(query_attention_weights.cpu().numpy(), cmap='hot', interpolation='nearest')
        #plt.colorbar()
        plt.title(f'Attention weights for query {j+1}')
        #plt.show()
        plt.savefig(os.path.join(param_dict['save_dir_model'], 'test',str(j)+name), bbox_inches='tight')
    pdb.set_trace()
    
    
    
    #attentions = all_attentions[k] #first image
    #num_tokens = attentions.shape[1]
    #depth = attentions.shape[0]
    
    #height = int(math.sqrt(num_tokens))
    #width = int(math.sqrt(num_tokens))

    #attentions = attentions.reshape(height, width, depth)

    #gray_att = torch.sum(attentions,2)
    #gray_att = gray_att / attentions.shape[2]

    #fig = plt.figure(figsize=(30, 50))
    #for i in range(1):#num_heads):
    #        a = fig.add_subplot(5, 4, i+1)
    #        imgplot = plt.imshow(gray_att.data.cpu().numpy())
    #        a.axis("off")
            #a.set_title(names[i].split('(')[0], fontsize=30)
        
     
    #plt.savefig(os.path.join(param_dict['save_dir_model'], name), bbox_inches='tight')
    


    

class global_var():
    def __init__(self):
        self.var = []

    def set_var(self, var):
        self.var = var

    def get_var(self):
        return self.var


class InferenceImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, transforms, g_vars):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.transforms = transforms
        

        sub_path = "test" 
        self.img_dir = os.path.join(self.root_dir, sub_path, "images")
        self.occ_label_dir = os.path.join(self.root_dir, sub_path, "occ_masks")
        self.visible_dir = os.path.join(self.root_dir, sub_path, "occ_labels")
        self.complete_dir = os.path.join(self.root_dir, sub_path, "labels")
        self.hidden_dir = os.path.join(self.root_dir, sub_path, "invisible")
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)
        
        # read occlusion annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.occ_label_dir):
            annotation_file_names.extend(files)
        self.occ_annotations = sorted(annotation_file_names)

        # read visible annotations
        vis_annotation_file_names = []
        for root, dirs, files in os.walk(self.visible_dir):
            vis_annotation_file_names.extend(files)
        self.vis_annotations = sorted(vis_annotation_file_names)

        # read hidden annotations
        hid_annotation_file_names = []
        for root, dirs, files in os.walk(self.hidden_dir):
            hid_annotation_file_names.extend(files)
        self.hid_annotations = sorted(hid_annotation_file_names)

        # read complete annotations
        gt_annotation_file_names = []
        for root, dirs, files in os.walk(self.complete_dir):
            gt_annotation_file_names.extend(files)
        self.complete_annotations = sorted(gt_annotation_file_names)
        

        
        g_vars.set_var(self.occ_annotations)
        
        
        assert len(self.images) == len(self.occ_annotations) == len(self.vis_annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)
    
    def get_annotations(self):
        return self.occ_annotations

    def __getitem__(self, idx):
        
        # RGB Images
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_path = os.path.join(self.img_dir, self.images[idx])

        # Occlusion annotation
        segmentation_map = cv2.imread(os.path.join(self.occ_label_dir, self.occ_annotations[idx]))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)
        segmentation_map = np.where(segmentation_map>0, 1, 0)
        occ_path = os.path.join(self.occ_label_dir, self.occ_annotations[idx])

        # Visible annotation
        v_segmentation_map = cv2.imread(os.path.join(self.visible_dir, self.vis_annotations[idx]))
        v_segmentation_map = cv2.cvtColor(v_segmentation_map, cv2.COLOR_BGR2GRAY)
        v_segmentation_map = np.where(v_segmentation_map>0, 1, 0)
        vis_path = os.path.join(self.visible_dir, self.vis_annotations[idx])

        # Hidden annotation
        hid_segmentation_map = cv2.imread(os.path.join(self.hidden_dir, self.hid_annotations[idx]))
        hid_segmentation_map = cv2.cvtColor(hid_segmentation_map, cv2.COLOR_BGR2GRAY)
        hid_segmentation_map = np.where(hid_segmentation_map>0, 1, 0)
        hid_path = os.path.join(self.hidden_dir, self.hid_annotations[idx])

        # Complete annotation
        com_segmentation_map = cv2.imread(os.path.join(self.complete_dir, self.complete_annotations[idx]))
        com_segmentation_map = cv2.cvtColor(com_segmentation_map, cv2.COLOR_BGR2GRAY)
        com_segmentation_map = np.where(com_segmentation_map>0, 1, 0)
        complete_path = os.path.join(self.complete_dir, self.complete_annotations[idx])

        
        sample = {'image': image, 'occ': segmentation_map, 'visible_mask': v_segmentation_map, 'gt':com_segmentation_map, 'hidden_mask': hid_segmentation_map}

        if self.transforms is not None:
            sample = self.transforms(sample)
            #encoded_inputs = self.feature_extractor(sample['image'], return_tensors="pt").pixel_values.cuda()
        encoded_inputs = self.feature_extractor(image, return_tensors="pt").pixel_values.cuda()
        
        encoded_inputs.squeeze_() # remove batch dimension
        
        #for k in encoded_inputs:
        #    encoded_inputs[k].squeeze_() # remove batch dimension
        
        sample['image'] = np.transpose(sample['image'], (2, 0, 1))
        sample['df_fimage'] = encoded_inputs
        sample['df_fooc'] = sample['occ']
        
        sample['img_path']= img_path
        sample['visible_path']= vis_path
        sample['occ_path']= occ_path
        sample['gt_path']= complete_path
        sample['hidden_path']= hid_path
        
       
        return sample
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
    n_batch["occ"] = inputs[2]
    n_batch["visible_mask"] = inputs[3]
    n_batch["hidden_mask"] = inputs[4]
    n_batch["img_path"] = inputs[5]
    n_batch["gt_path"] = inputs[6]
    n_batch["occ_path"] = inputs[7]
    n_batch["visible_path"] = inputs[8]
    n_batch["hidden_path"] = inputs[9]
    
    n_batch["pixel_values"] = inputs[10]
    n_batch["pro_target"] = inputs[11]
    
    if param_dict['use-fixed-model']: 
        n_batch['df_fimage'] = inputs[12]
        n_batch['df_fooc'] = inputs[13]

    

    #images = inputs[0]
    #segmentation_maps = inputs[1]

    #batch["original_images"] = inputs[2]
    #batch["original_segmentation_maps"] = inputs[3]

    #labels = [item["pro_target"] for item in batch]
    #batch[-1] = labels

    

    return n_batch

def test(testloader, model, epoch):

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

        checkpoint_path = os.path.join(param_dict['model_dir'], '360valiou_best.pth')  # load checkpoint
        state_dict = torch.load(checkpoint_path)['net']

        im_channel = 1
        if param_dict['inp_model'] == "MAT":
            G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=param_dict['img_size'], img_channels=im_channel) #Generator
            z = torch.randn(param_dict['batch_size'], 512).cuda()
        if param_dict['inp_model'] == "DFV2":
            G = DFV2_Generator(cnum_in=im_channel+2, cnum_out=im_channel, cnum=48, return_flow=False)
        
        model = torch.nn.DataParallel(G, device_ids=[0])
        model.load_state_dict(state_dict)
        epoch = torch.load(checkpoint_path)['epoch']
        print('epoch: ', epoch)
        
            
        model = model.cuda()
        model = model.eval()

    test_num = len(testloader.dataset)#.num_sample

    # To calculate metric of final window segmentation (complete)
    label_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    # To calculate metric of coarse window segmentation
    #predict_all_coarse = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    
    # To calculate metric of segmentation of occluders
    #label_all_occ = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    #predict_all_occ = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

 

    # To calculate metric of segmentation of visible windows
    #label_all_visible = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    #predict_all_visible = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    ind_dif = []
    complete = []
    incomplete = []
    

    if param_dict['two-steps']:
        label_all_mask = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_mask = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    
    if param_dict['three-steps'] or 1==1:
        label_all_mask = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_mask = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

        label_all_vis = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_vis = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

        # To calculate metric of segmentation of hidden windows
        label_all_hidden = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_hidden = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

        if os.path.exists(os.path.join(param_dict['pred_path'], 'hidden')) is False:
            os.makedirs(os.path.join(param_dict['pred_path'], 'hidden'))
    
    if os.path.exists(param_dict['pred_path']) is False:
        os.mkdir(param_dict['pred_path'])
    
    if param_dict['prob'] == True:
        if os.path.exists(param_dict['pred_prob_path']) is False:
            os.mkdir(param_dict['pred_prob_path'])

    #if os.path.exists(os.path.join(param_dict['pred_path'], 'visible')) is False:
    #    os.makedirs(os.path.join(param_dict['pred_path'], 'visible'))
    
    if os.path.exists(os.path.join(param_dict['pred_path'], 'occ')) is False:
        os.makedirs(os.path.join(param_dict['pred_path'], 'occ'))
    
    if param_dict['bayesian']:
        if os.path.exists(os.path.join(param_dict['pred_path'], 'unct_val_visual')) is False:
            os.makedirs(os.path.join(param_dict['pred_path'], 'unct_val_visual','slice'))
            os.makedirs(os.path.join(param_dict['pred_path'], 'unct_val_visual', 't_slice'))

    with torch.no_grad():
    #if 1==1:
        batch_num = 0
        n = 0

        # FOR effective receptive field computation
        #optimizer = create_optimizer_v2(model, 'adam', lr=param_dict['base_lr'])
        #meter = AverageMeter()
        #optimizer.zero_grad()

        for i, data in tqdm.tqdm(enumerate(testloader), ascii=True, desc="test step"):  # get data
            
            output_list = []

            if param_dict['amodal']: 
                images, labels, img_path, gt_path, modal = data['image'], data['visible_mask'], data['img_path'], data['visible_path'], data['occ']
                #images = torch.cat((images,modal), dim = 1)
                images = images+modal
            
            elif param_dict['two-steps']: 
                    images, labels, sf_fimages, occ_mask  = data['image'], data['gt'], data['df_fimage'], data['occ']

                    img_path = data['img_path'] # RGB image path
                    gt_path  = data['gt_path'] #Visible path
                    gt_path_mask  = data['occ_path'] # Occluder path

                    sf_fimages = sf_fimages.cuda()
                    #sf_focc_labels = sf_focc_labels.cuda()
                    #occ_mask = occ_mask.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    occ_mask = occ_mask

                    visible_mask = data['visible_mask']
                    hidden_mask = data['hidden_mask']
            
            elif mask2Former or param_dict['adversarial']:
                
                images = torch.stack(data["image"], dim=0)
                labels = torch.stack(data["gt"], dim=0)
                img_path = np.stack(data["img_path"])
                gt_path = np.stack(data["gt_path"])
                sf_fimages = torch.stack(data["df_fimage"], dim=0)
                sf_focc_labels = torch.stack(data["df_fooc"], dim=0)

                occ_mask = torch.stack(data["occ"], dim=0)
                visible_mask = torch.stack(data["visible_mask"], dim=0)
                hidden_mask = torch.stack(data["hidden_mask"], dim=0)

                #pro_target = data["pro_target"]
                    
                #for v in pro_target:                        
                #    v["boxes"] = v["boxes"].cuda() 
                   
                #for v in pro_target:                        
                #    v["class_labels"] = v["class_labels"].cuda()

                    
                sf_fimages = sf_fimages.cuda()
                sf_focc_labels = sf_focc_labels.cuda()
            

            else:

                
                loaded_data = data['image'], data['gt'], data['img_path'], data['gt_path'], data['occ'], data['occ_path'], data['visible_mask'], data['hidden_mask'], data['visible_path'], data['hidden_path']
                    

                images = loaded_data[0] #RGB image
                labels = loaded_data[1] #Window segmentation GT
                img_path = loaded_data[2] # RGB image path
                gt_path  = loaded_data[3] #Window segmentation GT path
                occ_mask = loaded_data[4] # Occluder image
                gt_path_mask  = loaded_data[5] # Occluder path                
                visible_mask = loaded_data[6] # Visible windows GT
                hidden_mask = loaded_data[7] # Hidden windows GT
                gt_path_vis = loaded_data[8] # Visible windows GT
                gt_path_hidden = loaded_data[9] # Hidden windows GT

                     
                if param_dict['three-steps']: 
                    sf_fimages = data['df_fimage']
                    sf_fimages = sf_fimages.cuda()

                    sf_focc_labels = data['df_fooc']
                    sf_focc_labels = sf_focc_labels.cuda()

                

                #images, labels, img_path, gt_path = data['image'], data['gt'], data['img_path'], data['gt_path']

            i += images.size()[0]
            
            images = images.cuda()
            

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

                occ_out = occ_model(pixel_values=sf_fimages, output_attentions=True)
                upsampled_logits = nn.functional.interpolate(occ_out.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                occ = upsampled_logits.argmax(dim=1) 
                out_occ = torch.unsqueeze(occ, 1) 
                
                
                #images = torch.cat((images,out_occ), dim=1).float()
                
                #images = (images+out_occ).float()

                
                #out_occ, outputs = model(sf_fimages,sf_focc_labels, images)
                outputs = model(images, out_occ)
            elif param_dict['three-steps']:
                comp_out = model(sf_fimages, sf_focc_labels, images.float())
                occ = comp_out[0]
                visible = comp_out[1]
                outputs = comp_out[2]

                ft_maps = comp_out[3]


            elif mask2Former or param_dict['adversarial']:
                
                visible = visi_model(images)

                occ = occ_model(sf_fimages, sf_focc_labels)
                occ = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                occ = occ.argmax(dim=1)  
                occ = torch.unsqueeze(occ, 1)
                    
                #outputs, add_out = model(images, occ.float())

                if param_dict['inp_model'] == 'MAT':
                    mask = 1 - occ.float()
                    outputs, stg1 = model(visible, mask, z, None, return_stg1=True)
                elif param_dict['inp_model'] == 'DFV2':
                    mask = occ.float()   
                    image_masked = visible * (1.-mask)  # mask image

                    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
                    x = torch.cat([image_masked, ones_x, ones_x*mask],dim=1)  # concatenate channels

                    _, x_stage2 = model(x, mask)

                    # complete image
                    outputs = visible * (1.-mask) + x_stage2 * mask

                
                
                #boxes = pos_process_boxes(add_out)
                
            else:
                #occ, coarse, outputs, visi = model(images)
                #occ, outputs = model(images)
                #comp_out = model(images)
                #outputs = comp_out[0]
                outputs = model(images)
                #outputs = out[0]
                #self_map = out[1][1] # Self-attention
                #cross_map = out[1][2] # Cross-attention
                
                
                
                
            
            ###########################################
            #USE LATER FOR EFFECTIVE RECEPTIVE FIELD COMPUTATION

            #if meter.count == 20:
                
                #np.save(os.path.join(param_dict['save_dir_model'],'ERF'), meter.avg)
                #exit()

            #samples = images.cuda(non_blocking=True)
            #samples.requires_grad = True
            #optimizer.zero_grad()
            #contribution_scores = get_input_grad(model, samples)

            #if img_path[0] == '/home/cero_ma/MCV/window_benchmarks/originals/resized/ecp-ref-occ60/test/images/monge_119bis.jpg':
            #    print('Here')
            #    meter.update(contribution_scores)
            #    np.save(os.path.join(param_dict['save_dir_model'],'ERF_119'), meter.avg)

            #if np.isnan(np.sum(contribution_scores)):
            #    print('got NAN, next image')
            #    continue
            #else:
            #    print('accumulate')
                #meter.update(contribution_scores)
            ###########################################


            
            if param_dict['bayesian']:
              unc_pred = unct
              unct_t = tools.utils.out2pred(unc_pred, param_dict['num_class'], 0.95, unct = True, thres= True)
              unct = tools.utils.out2pred(unct, param_dict['num_class'], param_dict['thread'], unct = True)


            if param_dict['prob']:

                pred_occ = tools.utils.out2pred(occ, param_dict['num_class'], param_dict['thread']) 
                pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'], prob= True)

            else:
                if param_dict['two-steps']:
                
                    pred_mask = tools.utils.out2pred(upsampled_logits, param_dict['num_class']+1, param_dict['thread'])
                
                if param_dict['three-steps']:
                
                    pred_mask = tools.utils.out2pred(occ, param_dict['num_class']+1, param_dict['thread'])
                    pred_vis = tools.utils.out2pred(visible, param_dict['num_class'], param_dict['thread'])
                    

                #pred_occ = tools.utils.out2pred(occ, param_dict['num_class'], param_dict['thread'])
                #pred_coarse = tools.utils.out2pred(coarse, param_dict['num_class'], param_dict['thread'])
                pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])
                

            batch_num += images.size()[0]
            for kk in range(len(img_path)):
                cur_name = os.path.basename(img_path[kk])
                
                pred_sub = pred[kk, :, :]
                
                label_all[n] = read_image(gt_path[kk], 'gt')
                predict_all[n]= pred_sub

                # BBox:

     
                #pred_sub_coarse = pred_coarse[kk, :, :]
                #predict_all_coarse[i] = pred_sub_coarse

                #pred_sub_occ = pred_occ[kk, :, :]
                #label_all_occ[n] = read_image(gt_path_mask[kk], 'gt')
                #predict_all_occ[n] = pred_sub_occ


                ###########################################

                #Visible prediction: complete - occluder mask
                
                pred_visible = pred_sub * (1-occ_mask[kk][0].numpy())
                label_all_vis[n] = visible_mask[kk][0].numpy()
                predict_all_vis[n] = pred_visible

                #Hidden prediction: complete - visible
                
                pred_hidden = np.where(pred_sub - pred_visible > 0, 1, 0) #(visible_mask[kk][0].numpy()) > 0, 1, 0) 
                label_all_hidden[n] = hidden_mask[kk][0].numpy() 
                predict_all_hidden[n] = pred_hidden 

                


                ###########################################
                
                
                #visualize_attention(cross_map, kk, 'att-'+cur_name.split('.')[0]+'.png')
                 
                
                if param_dict['bayesian']:
                    
                      unct_sub = unct[kk, :, :]
                      unct_t_sub = unct_t[kk, :, :]

                      yimage.io.write_image(
                        os.path.join(param_dict['pred_path'], 'unct_val_visual', 'slice', cur_name.split('.')[0]+'.png'),
                        unct_sub)#,
                        #color_table= [(0, 0, 0), (255, 255, 255)] ) #param_dict['color_table'])
                    
                      yimage.io.write_image(
                        os.path.join(param_dict['pred_path'], 'unct_val_visual', 't_slice', cur_name.split('.')[0]+'.png'),
                        unct_t_sub,
                        color_table=param_dict['color_table'])

                if param_dict['prob']:
                    yimage.io.write_image(
                        os.path.join(param_dict['pred_prob_path'], cur_name),
                        pred_sub)
                
                elif param_dict['two-steps']:
                        pred_sub_mask = pred_mask[kk, :, :]
                        label_all_mask[n] = read_image(gt_path_mask[kk], 'gt') 
                        predict_all_mask[n] = pred_sub_mask

                        yimage.io.write_image(
                        os.path.join(param_dict['pred_path'], 'occ', cur_name.split('.')[0]+'.png'),
                        pred_sub_mask,
                        color_table=param_dict['color_table'])

                elif param_dict['three-steps']:
                        pred_sub_mask = pred_mask[kk, :, :]
                        label_all_mask[n] = read_image(gt_path_mask[kk], 'gt') 
                        predict_all_mask[n] = pred_sub_mask

                        yimage.io.write_image(
                        os.path.join(param_dict['pred_path'], 'occ', cur_name.split('.')[0]+'.png'),
                        pred_sub_mask,
                        color_table=param_dict['color_table'])

                        pred_sub_vis = pred_vis[kk, :, :]
                        label_all_vis[n] = read_image(gt_path_vis[kk], 'gt') 
                        predict_all_vis[n] = pred_sub_vis

                        yimage.io.write_image(
                        os.path.join(param_dict['pred_path'], 'occ', cur_name.split('.')[0]+'.png'),
                        pred_sub_mask,
                        color_table=param_dict['color_table'])
                        
                        # Extract only segmentation of hidden area
                        pred_hidden = np.where(pred_sub - pred_sub_vis > 0, 1, 0) #(visible_mask[kk][0].numpy()) > 0, 1, 0) 
                        label_all_hidden[n] = hidden_mask[kk][0].numpy() 
                        predict_all_hidden[n] = pred_hidden 

                        yimage.io.write_image(
                        os.path.join(param_dict['pred_path'], 'hidden', cur_name.replace('.jpg','.png')),
                        pred_hidden,
                        color_table=param_dict['color_table'])
                
                #yimage.io.write_image(
                img = (pred_sub*255).astype(float)
                #cv2.rectangle(img, start_point, end_point, color=(0,255,0), thickness=2)
                cv2.imwrite(
                        os.path.join(param_dict['pred_path'], cur_name.replace('.jpg','.png')),
                        img)
                        #color_table=param_dict['color_table'])
                
                    #yimage.io.write_image(
                    #    os.path.join(param_dict['pred_path'], 'visible', cur_name.replace('.jpg','.png')),
                    #    pred_visible,
                    #    color_table=param_dict['color_table'])
                    
                    
 
                    #yimage.io.write_image(
                    #    os.path.join(param_dict['pred_path'], 'coarse', cur_name.replace('.jpg','.png')),
                    #    pred_sub_coarse,
                    #    color_table=param_dict['color_table'])  

                    #yimage.io.write_image(
                    #    os.path.join(param_dict['pred_path'], 'occ', cur_name.replace('.jpg','.png')),
                    #    pred_sub_occ,
                    #    color_table=param_dict['color_table'])                  
                
                # Compute feature maps of visible regions -> TODO: to find features of complete masks....
                
                #ind_iou = jaccard_score(pred_sub_vis, labels[kk], average ='micro') #Here compare with complete GT: labels
                #ind_dif.append([ind_iou, gt_path[kk].split('/')[-1]])
                #if ind_dif[n][0] >= 0.68:                    
                #    complete.append(ft_maps[kk])
                #else:
                #    incomplete.append(ft_maps[kk])

                n += 1
                
       
        first_elements = [sub_array[0] for sub_array in ind_dif]

        processed = []
        for feature_map in complete:
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        
        
        fig = plt.figure(figsize=(30, 50))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i+1)
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            #a.set_title(names[i].split('(')[0], fontsize=30)
        plt.savefig(os.path.join(param_dict['pred_path'], 'featMapL5_complete.jpg'), bbox_inches='tight')

        in_processed = []
        for feature_map in incomplete:
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]
            in_processed.append(gray_scale.data.cpu().numpy())
               
        fig = plt.figure(figsize=(30, 50))
        for i in range(len(in_processed)):
            a = fig.add_subplot(5, 4, i+1)
            imgplot = plt.imshow(in_processed[i])
            a.axis("off")
            #a.set_title(names[i].split('(')[0], fontsize=30)
        plt.savefig(os.path.join(param_dict['pred_path'], 'featMapL5_incomplete.jpg'), bbox_inches='tight')
    
    
  
        
        

                
                    
                        
        print('Segmentation')
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            param_dict['save_dir_model'],
            file_name =str(epoch)+'_accuracy.txt')
        
        if param_dict['two-steps']:
            print('Occlusion')
            precision_mask, recall_mask, f1ccore_mask, OA_mask, IoU_mask, mIOU_mask = get_acc_v2(
                label_all_mask, predict_all_mask,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model']),
                file_name =str(epoch)+'_occlusion_mask.txt')
        
        if param_dict['three-steps']:
            print('Occlusion')
            precision_mask, recall_mask, f1ccore_mask, OA_mask, IoU_mask, mIOU_mask = get_acc_v2(
                label_all_mask, predict_all_mask,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model']),
                file_name =str(epoch)+'_occlusion_mask.txt')
            
        print('Visible windows')
        results_vis = get_acc_v2(
                label_all_vis, predict_all_vis,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model']),
                file_name =str(epoch)+'_visible.txt')

        print('Hidden windows')
        results_hidden = get_acc_v2(
                label_all_hidden, predict_all_hidden,
                param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
                os.path.join(param_dict['save_dir_model']),
                file_name =str(epoch)+'_hidden.txt')
            
        
        #precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
        #    label_all, predict_all_coarse,
        #    param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
        #    os.path.join(param_dict['pred_path'], 'coarse'))
        
        #print('Occlusion')
        #precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
        #    label_all_occ, predict_all_occ,
        #    param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
        #    os.path.join(param_dict['pred_path'], 'occ'))
       
        #print('Visible windows')
        #precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
        #    label_all_visible, predict_all_visible,
        #    param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
        #    os.path.join(param_dict['pred_path'], 'visible')) 

        #print('Hidden windows')
        #precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
        #    label_all_hidden, predict_all_hidden,
        #    param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
        #    os.path.join(param_dict['pred_path'], 'hidden'))
        
      


def load_model(model_path):

    """ if param_dict['fine-tune-DL'] == True:
        model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=8) #8 or 16??? deeplabv3plus_mobilenet
        num_classes = param_dict['num_class']
        last_layer = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) 
        model.classifier.classifier[3] = last_layer
        #model = torch.nn.DataParallel(model, device_ids=gpu_list)
        model = torch.nn.DataParallel(model, device_ids=[0])
        state_dict = torch.load(model_path)['net']
        new_state_dict = OrderedDict()
        model.load_state_dict(state_dict)
        print('epoch: ', torch.load(model_path)['epoch'])
        model.cuda()
        model.eval()
        return model """

    if param_dict['amodal']:
        input_bands = param_dict['input_bands']+1
    else:
        input_bands = param_dict['input_bands']

    if param_dict['fine-tune-DL']:
        print('DeepLabV3+')
        model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=8) #8 or 16??? deeplabv3plus_mobilenet
        #network.convert_to_separable_conv(model.classifier)
        #set_bn_momentum(model.backbone, momentum=0.01)
        num_classes = param_dict['num_class']
        last_layer = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) 
            
        model.classifier.classifier[3] = last_layer   
         
    else:
        model = get_net(param_dict['model_name'], input_bands, param_dict['num_class'],
                    param_dict['img_size'], param_dict['pretrained_model'])

    #model = torch.nn.DataParallel(model, device_ids=[0])
    
    
    state_dict = torch.load(model_path)['net']
    new_state_dict = OrderedDict()
    #for k, v in state_dict.items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    
    model.load_state_dict(state_dict)
    
    epoch = torch.load(model_path)['epoch']
    print('epoch: ', epoch)
    #pdb.set_trace()
    model.cuda()
    model.eval()
    return model, epoch


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

    mask2Former = False

    model_path = os.path.join(param_dict['model_dir'], '360valiou_best.pth')#'valiou_best.pth')#0valiou_best.pth')2201
    
    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor()])  # data pocessing and data augumentation

    # Transformations for Occlusion training with Segoformer
    sf_transform = aug.Compose([
        aug.Flip(p=0.5),
        #aug.RandomCrop(width=128, height=128), does not work for this dataset and task
        aug.HorizontalFlip(p=0.5),
        aug.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        aug.geometric.rotate.Rotate (limit=[-15, 15]) ])
    
    if param_dict['two-steps'] or param_dict['three-steps']:
        feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
        road_test = AmodalSegmentation(txt_path=param_dict['test_list'], transform=composed_transforms_val,  occSegFormer = True, inference = True, feature_extractor = feature_extractor_inference)  # get data
    elif mask2Former or param_dict['adversarial']:
        feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
        processor = DetrImageProcessor(do_resize =False, do_rescale = False)
        road_test = AmodalSegmentation(txt_path=param_dict['test_list'], transform=composed_transforms_val, sf_transform=sf_transform, occSegFormer = True, feature_extractor= feature_extractor,  processor=processor) 
    else:
        road_test = AmodalSegmentation(txt_path=param_dict['test_list'], transform=composed_transforms_val)  # get data
    

    
    
    testloader = DataLoader(road_test, batch_size=param_dict['batch_size'], shuffle=False,
                           num_workers=param_dict['num_workers'], drop_last=False, collate_fn=collate_fn)  # define traindata  


    root_dir = "/home/cero_ma/MCV/window_benchmarks/originals/resized/ecp-ref-occ60/"
    
    transform = standard_transforms.Compose([
        
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std'])]) 

    #feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
    #g_vars = global_var()
    
    #if param_dict['three-steps']: #Inferece using SegFormer
    #    test_dataset = InferenceImageSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor_inference, transforms = transform, g_vars= g_vars)
    #    annotations = g_vars.get_var()
    #    testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    if not param_dict['adversarial']:
        model, epoch = load_model(model_path)
        test(testloader, model, epoch)
    else:
        test(testloader, None, None)

    

    
