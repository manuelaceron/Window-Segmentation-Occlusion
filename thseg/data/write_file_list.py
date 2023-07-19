import os, glob
import random
import sys
import pdb

src_path = r'/home/cero_ma/MCV/window_benchmarks/originals/resized/ecp-ref-occ60/'



state = 'test'
amodal = True #training input: image + visible mask

image_path = os.path.join(src_path, state, 'images')
label_path = os.path.join(src_path, state, 'labels')
occ_path = os.path.join(src_path, state, 'occ_masks')
visible_path = os.path.join(src_path, state, 'occ_labels')
hidden_path = os.path.join(src_path, state, 'invisible')

##############CITYSCAPES###################

""" label_src_path = '/home/cero_ma/MCV/cityscapes/test_multiclass_mask/gtFine/'
label_path = os.path.join( label_src_path , state , "*" , "*.png" )
labels = glob.glob( label_path )

img_src_path = '/home/cero_ma/MCV/cityscapes/leftImg8bit/'
image_path = os.path.join( img_src_path , state , "*" , "*.png" )
images = glob.glob( image_path )

with open('{}_list_cityscapes_multi.txt'.format(state), 'w') as ff:
    for name in images:
       
        cur_info = '{}  {}\n'.format(name, name.replace(img_src_path, label_src_path))
        ff.writelines(cur_info)  """
###########################################




if amodal:
    modal_path = "/home/cero_ma/MCV/window_benchmarks/originals/resized/graz-ref-occ60/test/occ_masks/" #'/home/cero_ma/MCV/code220419_windows/0401_files/Res_UNet_50_Pretrained_400e_oxford_robotcar_modal/ecp-occ60_inference/val/'#

images = os.listdir(image_path)
random.shuffle(images)

#artdeco, graz: png
#ECP, CMP: jpg

with open('{}_ecp.txt'.format(state), 'a') as ff:
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
                #cur_info = '{},'.format(os.path.join(image_path, name))
                #cur_info = '{},'.format(os.path.join(label_path, name_label))  
            else:
                cur_info = '{}  {}  {}  {}  {}\n'.format(os.path.join(image_path, name), os.path.join(label_path, name_label), os.path.join(occ_path, name_modal), 
                os.path.join(visible_path, name_label), os.path.join(hidden_path, name_label)) 

            ff.writelines(cur_info)

