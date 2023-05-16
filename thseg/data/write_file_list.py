import os
import random
import sys

src_path = r'/home/cero_ma/MCV/window_benchmarks/originals/resized/ecp-ref-occ60/'

state = 'train'
amodal = True #training input: image + visible mask

image_path = os.path.join(src_path, state, 'images')
label_path = os.path.join(src_path, state, 'labels')
if amodal:
    modal_path = "/home/cero_ma/MCV/window_benchmarks/originals/resized/ecp-ref-occ60/train/occ_labels/" #'/home/cero_ma/MCV/code220419_windows/0401_files/Res_UNet_50_Pretrained_400e_oxford_robotcar_modal/ecp-occ60_inference/val/'#

images = os.listdir(image_path)
random.shuffle(images)

#artdeco, graz: png
#ECP, CMP: jpg

with open('{}_list_finetune.txt'.format(state), 'w') as ff:
    for name in images:
        if name.split('.')[1] == 'jpg' :
            name_label = name.replace('jpg', 'png')
            name_modal = name.replace('jpg', 'png') #comment for ecp oxford
        else:
            name_label = name
            name_modal = name

        if os.path.exists(os.path.join(image_path, name)) is True and os.path.exists(os.path.join(label_path, name_label)) is True:
        
            if not amodal:
                cur_info = '{}  {}\n'.format(os.path.join(image_path, name), os.path.join(label_path, name_label)) 
            else:
                cur_info = '{}  {}  {}\n'.format(os.path.join(image_path, name), os.path.join(label_path, name_label), os.path.join(modal_path, name_modal)) 

            ff.writelines(cur_info)

