from __future__ import division
import sys
import os
import torchvision
from tools.utils import read_image
import yimage
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
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import albumentations as aug
from timm.utils import AverageMeter
from timm.optim import create_optimizer_v2
from sklearn.metrics import jaccard_score
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

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

        
        sample = {'image': image, 'occ': segmentation_map, 'visible_mask': v_segmentation_map}

        if self.transforms is not None:
            sample = self.transforms(sample)
            #encoded_inputs = self.feature_extractor(sample['image'], return_tensors="pt").pixel_values.cuda()
        encoded_inputs = self.feature_extractor(image, return_tensors="pt").pixel_values.cuda()
        
        encoded_inputs.squeeze_() # remove batch dimension
        
        #for k in encoded_inputs:
        #    encoded_inputs[k].squeeze_() # remove batch dimension
        
        sample['image'] = np.transpose(sample['image'], (2, 0, 1))
        sample['df_fimage'] = encoded_inputs
        
        sample['img_path']= img_path
        sample['visible_path']= vis_path
        sample['occ_path']= occ_path
        
       
        return sample

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
    
    test_num = len(testloader.dataset)#.num_sample

    # To calculate metric of final window segmentation (complete)
    label_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    # To calculate metric of coarse window segmentation
    #predict_all_coarse = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    
    # To calculate metric of segmentation of occluders
    #label_all_occ = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    #predict_all_occ = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    # To calculate metric of segmentation of hidden windows
    #label_all_hidden = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    #predict_all_hidden = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    # To calculate metric of segmentation of visible windows
    #label_all_visible = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    #predict_all_visible = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)

    ind_dif = []
    complete = []
    incomplete = []
    

    if param_dict['two-steps']:
        label_all_mask = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
        predict_all_mask = np.zeros((test_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    
    if os.path.exists(param_dict['pred_path']) is False:
        os.mkdir(param_dict['pred_path'])
    
    if param_dict['prob'] == True:
        if os.path.exists(param_dict['pred_prob_path']) is False:
            os.mkdir(param_dict['pred_prob_path'])

    #if os.path.exists(os.path.join(param_dict['pred_path'], 'visible')) is False:
    #    os.makedirs(os.path.join(param_dict['pred_path'], 'visible'))
    #if os.path.exists(os.path.join(param_dict['pred_path'], 'hidden')) is False:
    #    os.makedirs(os.path.join(param_dict['pred_path'], 'hidden'))
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
            
            elif param_dict['two-steps'] == True: 
                    images, labels, sf_fimages, gt_occ  = data['image'], data['visible_mask'], data['df_fimage'], data['occ']

                    img_path = data['img_path'] # RGB image path
                    gt_path  = data['visible_path'] #Visible path
                    gt_path_mask  = data['occ_path'] # Occluder path

                    sf_fimages = sf_fimages.cuda()
                    #sf_focc_labels = sf_focc_labels.cuda()
                    gt_occ = gt_occ.view(images.size()[0], param_dict['img_size'], param_dict['img_size']).long() 
                    gt_occ = gt_occ.cuda()
            else:
                loaded_data = data['image'], data['gt'], data['img_path'], data['gt_path'], data['occ_path'], data['occ'], data['visible_mask'], data['hidden_mask'], data['visible_path'], data['hidden_path']
                

                images = loaded_data[0] #RGB image
                labels = loaded_data[1] #Window segmentation GT
                img_path = loaded_data[2] # RGB image path
                gt_path  = loaded_data[8] #Window segmentation GT path
                occ_mask = loaded_data[5] # Occluder image
                gt_path_mask  = loaded_data[4] # Occluder path                
                visible_mask = loaded_data[6] # Visible windows GT
                hidden_mask = loaded_data[7] # Hidden windows GT
                visible_mask_path = loaded_data[8] # Visible windows GT
                hidden_mask_path = loaded_data[9] # Hidden windows GT

                

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

                occ = occ_model(pixel_values=sf_fimages)
                upsampled_logits = nn.functional.interpolate(occ.logits, size=param_dict['img_size'], mode="bilinear", align_corners=False) 
                occ = upsampled_logits.argmax(dim=1) 
                out_occ = torch.unsqueeze(occ, 1) 
                
                
                #images = torch.cat((images,out_occ), dim=1).float()
                
                images = (images+out_occ).float()

                
                #out_occ, outputs = model(sf_fimages,sf_focc_labels, images)
                outputs = model(images)

            else:
                #occ, coarse, outputs, visi = model(images)
                #occ, outputs = model(images)
                comp_out = model(images)
                outputs = comp_out[0]
                
            
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
                    

                #pred_occ = tools.utils.out2pred(occ, param_dict['num_class'], param_dict['thread'])
                #pred_coarse = tools.utils.out2pred(coarse, param_dict['num_class'], param_dict['thread'])
                pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])
                

            batch_num += images.size()[0]
            for kk in range(len(img_path)):
                cur_name = os.path.basename(img_path[kk])
                
                pred_sub = pred[kk, :, :]
                

                ind_iou = jaccard_score(pred_sub, labels[kk][0], average ='micro')
                
                ind_dif.append([ind_iou, gt_path[kk].split('/')[-1]])
                
                if ind_dif[n][0] >= 0.68:                    
                    complete.append(comp_out[5][kk])
                else:
                    incomplete.append(comp_out[5][kk])
                    

                
                label_all[n] = read_image(gt_path[kk], 'gt')
                predict_all[n]= pred_sub
     
                #pred_sub_coarse = pred_coarse[kk, :, :]
                #predict_all_coarse[i] = pred_sub_coarse

                #pred_sub_occ = pred_occ[kk, :, :]
                #label_all_occ[n] = read_image(gt_path_mask[kk], 'gt')
                #predict_all_occ[n] = pred_sub_occ


                ###########################################

                #Visible prediction: complete - occluder mask
                
                #pred_visible = pred_sub * (1-occ_mask[kk][0].numpy())
                #label_all_visible[n] = visible_mask[kk][0].numpy()
                #predict_all_visible[n] = pred_visible

                #Hidden prediction: complete - visible
                
                #pred_hidden = np.where(pred_sub - pred_visible > 0, 1, 0) #(visible_mask[kk][0].numpy()) > 0, 1, 0) 
                #label_all_hidden[n] = hidden_mask[kk][0].numpy() 
                #predict_all_hidden[n] = pred_hidden 

                


                ###########################################

                
                
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
                
                yimage.io.write_image(
                        os.path.join(param_dict['pred_path'], cur_name.replace('.jpg','.png')),
                        pred_sub,
                        color_table=param_dict['color_table'])
                
                    #yimage.io.write_image(
                    #    os.path.join(param_dict['pred_path'], 'visible', cur_name.replace('.jpg','.png')),
                    #    pred_visible,
                    #    color_table=param_dict['color_table'])
                    
                    #yimage.io.write_image(
                    #    os.path.join(param_dict['pred_path'], 'hidden', cur_name.replace('.jpg','.png')),
                    #    pred_hidden,
                    #    color_table=param_dict['color_table'])
 
                    #yimage.io.write_image(
                    #    os.path.join(param_dict['pred_path'], 'coarse', cur_name.replace('.jpg','.png')),
                    #    pred_sub_coarse,
                    #    color_table=param_dict['color_table'])  

                    #yimage.io.write_image(
                    #    os.path.join(param_dict['pred_path'], 'occ', cur_name.replace('.jpg','.png')),
                    #    pred_sub_occ,
                    #    color_table=param_dict['color_table'])                  

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
        plt.savefig(str('/home/cero_ma/MCV/code220419_windows/0401_files/Res_UNet_50_ecp_visible/featMapL5_complete.jpg'), bbox_inches='tight')

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
        plt.savefig(str('/home/cero_ma/MCV/code220419_windows/0401_files/Res_UNet_50_ecp_visible/featMapL5_incomplete.jpg'), bbox_inches='tight')
    
    
  
        
        

                
                    
                        
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
    
    #pdb.set_trace()
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

    model_path = os.path.join(param_dict['model_dir'], '220valiou_best.pth')#'valiou_best.pth')#0valiou_best.pth')2201
    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor()])  # data pocessing and data augumentation

    
    if param_dict['two-steps']:
        feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
        road_test = AmodalSegmentation(txt_path=param_dict['test_list'], transform=composed_transforms_val,  occSegFormer = True, inference = True, feature_extractor = feature_extractor_inference)  # get data
    else:
        road_test = AmodalSegmentation(txt_path=param_dict['test_list'], transform=composed_transforms_val)  # get data
    
    
    testloader = DataLoader(road_test, batch_size=param_dict['batch_size'], shuffle=False,
                           num_workers=param_dict['num_workers'], drop_last=False)  # define traindata  


    root_dir = "/home/cero_ma/MCV/window_benchmarks/originals/resized/ecp-ref-occ60/"
    
    transform = standard_transforms.Compose([
        
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std'])]) 

    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
    g_vars = global_var()
    

    #test_dataset = InferenceImageSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor_inference, transforms = transform, g_vars= g_vars)
    #annotations = g_vars.get_var()
    #testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)


    model, epoch = load_model(model_path)


    test(testloader, model, epoch)
