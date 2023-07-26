from __future__ import print_function, division
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tools import utils
import pdb, json




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
                 feature_extractor = None,
                 sf_transform = None,
                 processor = None
                 ):
        self.list_sample = open(txt_path, 'r').readlines()
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        
        print('# samples: {}'.format(self.num_sample))
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.occSegFormer = occSegFormer
        self.inference = inference

        self.processor = processor
        
        ann_file = "/home/cero_ma/MCV/code220419_windows/thseg/runs/labelme2coco/ecp-refined-coco.json"
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        self.ann_folder = "/home/cero_ma/MCV/window_benchmarks/originals/resized/ecp-ref-occ60/train/labels/"
        
        

    def __len__(self):
        return self.num_sample
        # return 256

    def __getitem__(self, index):
        
        
        
        _img, _target, img_path, gt_path, _occ, occ_path, _hidden, hidden_path, _visible, visible_path = self._make_img_gt_point_pair(index) 
        
        
        #annotation: image_id (image), id (object id)
        img_name = img_path.split('/')[-1].split('.')[0]
        ann_name = self.coco['images'][index]['file_name'].split('\\')[-1].split('.')[0]

        the_id = [item for item in self.coco['images'] if img_name in item['file_name']][0]['id']

        ann_info = {'image_id': the_id,
            'annotations': [item for item in self.coco['annotations'] if item['image_id'] == the_id],
            'images': [item for item in self.coco['images'] if img_name in item['file_name']]}

        
        #ann_info.update(resize_annotation(ann_info, (ann_info['images'][0]['height'],ann_info['images'][0]['width']), (_img.shape[0], _img.shape[1]) ) )
        grid = draw_boundaries(ann_info)

        #cv2.imshow('Grid Overlay', ((_visible+grid)*255).astype(float))
    
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #pdb.set_trace()

        encoding = self.processor(images=_img, annotations=ann_info, masks_path=self.ann_folder, return_tensors="pt")
        
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        pro_target = encoding["labels"][0]
        #ins_tg = generate_instance_masks(pro_target)
        #ins_tg = np.transpose(ins_tg, (1,2,0))
        
        

        sample = {'image': _img, 'gt': _target, 'img_sf': _img, 'occ': _occ, 'occ_sf': _occ, 'visible_mask': _visible, 'hidden_mask': _hidden, 'grid': grid}
        
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        
        sample['img_path'] = img_path
        sample['gt_path'] = gt_path
        sample['occ_path'] = occ_path
        sample['visible_path'] = visible_path
        sample['hidden_path'] = hidden_path
        sample['pixel_values'] = pixel_values
        sample['pro_target'] = pro_target

        

        #TODO: Instead of pixel_values (1 to all real pixels, I need pixel values from occlusion mask, )
        

        #pdb.set_trace()
        if self.occSegFormer:

            #if self.inference: # No transformation
            #    sf_img= _img
            #else: #Sf transformations
            #    pdb.set_trace()
            #    augmented = self.sf_transform(image=_img, mask=_occ)
            #    sf_img= augmented['image']
            #    sf_occ = augmented['mask']

            #sf_img= _img       
            #sf_occ = _occ #sample['occ'].numpy() #GT occlusions
            #sf_img = np.transpose(sf_img, (1, 2, 0))
            #sf_occ = np.squeeze(sf_occ, 0)

            
            if self.inference:
                encoded_inputs = self.feature_extractor(_img, return_tensors="pt").pixel_values
                encoded_inputs.squeeze_()
                sample['df_fimage'] = encoded_inputs
            else:
                
                encoded_inputs = self.feature_extractor(sample['img_sf'], sample['occ_sf'], return_tensors="pt")
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


        #json_name=img_path.split('/')[-1].split('.')[0]+'.json'
        #mode = img_path.split('/')[-3]
        #json_root = "/home/cero_ma/MCV/window_benchmarks/originals/split-json-rectified/coco/ecp/"
        #json_path = os.path.join(json_root,mode,json_name)
        
        #with open(json_path, 'r') as f:
        #    ann_json = json.load(f)
        
        #ann_json = extract_json_info(ann_json)
        
        
        
        return _img, _target, img_path, gt_path, _occ, occ_path, _hidden, hidden_path, _visible, visible_path#, ann_json

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

def extract_json_info(ann_file):
    
    ori_h = ann_file['imageHeight']
    ori_w =ann_file['imageWidth']
    target_h = target_w = 512

    window_objects = []
    out = []
    max_windows = 400
    
    for shape in ann_file['shapes']:
        if shape['label'] == 'window':
            points = shape['points']
            original_coords = [
                points[0][0],
                points[0][1],
                points[1][0],
                points[1][1]
            ]
            resized_coords = resize_coordinates(original_coords, ori_h, ori_w, target_h, target_w)
            window_objects.append(resized_coords)
   
    padded_coordinates = torch.zeros((max_windows, 4))

    for i, coord in enumerate(window_objects):
        padded_coordinates[i] = torch.tensor(coord)

    
    print('padded_coordinates', len(padded_coordinates))
    return padded_coordinates
    
    
def resize_coordinates(coords, original_height, original_width, new_height, new_width):
    
    ratio_width = new_width / original_width
    ratio_height = new_height / original_height

    resized_coords = [
        int(coords[0] * ratio_width),
        int(coords[1] * ratio_height),
        int(coords[2] * ratio_width),
        int(coords[3] * ratio_height)
    ]

    return resized_coords

def resize_annotation(annotation, orig_size, target_size):

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios
    
    for key, value in annotation.items():
        
        if key == 'annotations':            
            for val in value:
                
                
                boxes = val['bbox']
                scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
                val['bbox'] =  scaled_boxes

                
                area = val['area']
                scaled_area = area * (ratio_width * ratio_height)
                val['area'] = scaled_area

                val['size'] = target_size
                
                # Object class
                #if val['category_id'] == 0:
                #    val['category_id'] = 1
                #else:
                #    val['category_id'] = val['category_id']


    
             
    return annotation           

def generate_instance_masks(ann_data):

    ann = ann_info['annotations']
    num_boxes= len(ann)
    image_size = (num_boxes,512,512)
    
    H = ann_info['images'][0]['height']
    W = ann_info['images'][0]['width']

    
    num_boxes= len(ann)
    ins_mask = np.zeros((num_boxes,H,W))

    for i in range(num_boxes):
        x, y, width, height = ann[i]

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + width), int(y + height)

        ins_mask[i, y1:y2, x1:x2] = 1.0
         
    re_ins_mask = cv2.resize(ins_mask, image_size, interpolation=cv2.INTER_NEAREST)

    return ins_mask

def draw_boundaries(ann_info):
    
    
    ann = ann_info['annotations']
    num_boxes= len(ann)
    image_size = (512,512)
    
    H = ann_info['images'][0]['height']
    W = ann_info['images'][0]['width']

    grid = np.zeros((H,W))
    grid_color = 1

    for i in range(num_boxes):
        x, y, width, height = ann[i]['bbox']
        x=int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        try:
            cv2.line(grid, (x, 0), (x,H), grid_color, 1)
            cv2.line(grid, (x+width, 0), (x+width,H), grid_color, 1)
            cv2.line(grid, (0, y), (W,y), grid_color, 1)
            cv2.line(grid, (0, y+height), (W,y+height), grid_color, 1)
        except:
            pdb.set_trace()
    
    re_grid = cv2.resize(grid, image_size, interpolation=cv2.INTER_NEAREST)
    
    
    
    return re_grid
    