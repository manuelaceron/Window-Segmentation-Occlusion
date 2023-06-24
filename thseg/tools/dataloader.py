from __future__ import print_function, division
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from tools import utils
import pdb




class IsprsSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 txt_path=None,
                 transform=None
                 ):
        self.list_sample = open(txt_path, 'r').readlines()
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        
        print('# samples: {}'.format(self.num_sample))
        self.transform = transform

    def __len__(self):
        return self.num_sample
        # return 256

    def __getitem__(self, index):
        
        _img, _target, img_path, gt_path = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['img_path'] = img_path
        sample['gt_path'] = gt_path
        return sample

    def _make_img_gt_point_pair(self, index):
        file = self.list_sample[index].strip()
        img_path = file.split('  ')[0]
        gt_path = file.split('  ')[1]

        _img = utils.read_image(os.path.join(img_path))
        
        _target = utils.read_image(os.path.join(gt_path), 'gt').astype(np.int32)
        
        return _img, _target, img_path, gt_path

class AmodalSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 txt_path=None,
                 transform=None,
                 occSegFormer = False,
                 inference = False,
                 feature_extractor = None

                 ):
        self.list_sample = open(txt_path, 'r').readlines()
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        
        print('# samples: {}'.format(self.num_sample))
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.occSegFormer = occSegFormer
        self.inference = inference
        

    def __len__(self):
        return self.num_sample
        # return 256

    def __getitem__(self, index):
        _img, _target, img_path, gt_path, _occ, occ_path, _hidden, hidden_path, _visible, visible_path = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target, 'occ': _occ, 'visible_mask': _visible, 'hidden_mask': _hidden}
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        
        sample['img_path'] = img_path
        sample['gt_path'] = gt_path
        sample['occ_path'] = occ_path
        sample['visible_path'] = visible_path
        sample['hidden_path'] = hidden_path

        #pdb.set_trace()
        if self.occSegFormer:
            sf_img= sample['image'].numpy()
            sf_occ = sample['occ'].numpy() #GT occlusions
            sf_img = np.transpose(sf_img, (1, 2, 0))
            sf_occ = np.squeeze(sf_occ, 0)
            
            if self.inference:
                encoded_inputs = self.feature_extractor(sf_img, return_tensors="pt").pixel_values
                
                encoded_inputs.squeeze_()
                sample['df_fimage'] = encoded_inputs
            else:
                encoded_inputs = self.feature_extractor(sf_img, sf_occ, return_tensors="pt")
                for k,v in encoded_inputs.items():
                    encoded_inputs[k].squeeze_() # remove batch dimension
                
                sample['df_fimage'] = encoded_inputs['pixel_values']
                sample['df_fooc'] = encoded_inputs['labels']
               
            
            return sample
        
        return sample

    def _make_img_gt_point_pair(self, index):
        file = self.list_sample[index].strip()
        
        
        img_path = file.split('  ')[0] #RGB image
        gt_path = file.split('  ')[1] #GT WSeg
        occ_path = file.split('  ')[2] #GT occluder
        visible_path = file.split('  ')[3] #GT visible windows
        hidden_path = file.split('  ')[4] #GT hidden windows
        
        
        _img = utils.read_image(os.path.join(img_path))
        _target = utils.read_image(os.path.join(gt_path), 'gt').astype(np.int32)
        _occ = utils.read_image(os.path.join(occ_path), 'gt').astype(np.int32)
        
        
        _hidden = utils.read_image(os.path.join(hidden_path), 'am').astype(np.int32)
        
        _visible = utils.read_image(os.path.join(visible_path), 'am').astype(np.int32)
        
        return _img, _target, img_path, gt_path, _occ, occ_path, _hidden, hidden_path, _visible, visible_path

class InferenceSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 txt_path=None,
                 transform=None
                 ):
        self.list_sample = open(txt_path, 'r').readlines()
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        
        print('# samples: {}'.format(self.num_sample))
        self.transform = transform

    def __len__(self):
        return self.num_sample
        # return 256

    def __getitem__(self, index):
        
        _img, img_path, folder = self._make_img_gt_point_pair(index)
        sample = {'image': _img}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['img_path'] = img_path
        sample['folder'] = folder
        return sample

    def _make_img_gt_point_pair(self, index):
        file = self.list_sample[index].strip()
        img_path = file.split('  ')[0]
        folder = file.split('  ')[1]
        _img = utils.read_image(os.path.join(img_path))
        
        
        return _img, img_path, folder 

