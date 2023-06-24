import os
import random
import sys

src_path = r'/home/cero_ma/MCV/window_benchmarks/originals/resized/artdeco/'

state = 'inference'

train_image_path = os.path.join(src_path, 'train', 'images')
val_image_path = os.path.join(src_path, 'val', 'images')
test_image_path = os.path.join(src_path, 'test', 'images')

train_images = os.listdir(train_image_path)
val_images = os.listdir(val_image_path)
test_images = os.listdir(test_image_path)

#random.shuffle(images)

#artdeco, graz: png
#ECP, CMP: jpg

with open('{}_list.txt'.format(state), 'w') as ff:
    for name in train_images:
        if name.split('.')[1] == 'jpg' :
            name_label = name.replace('jpg', 'png')
        else:
            name_label = name

        cur_info = '{}  {}\n'.format(os.path.join(train_image_path, name), 'train') 
        ff.writelines(cur_info)
    
    for name in val_images:
        if name.split('.')[1] == 'jpg' :
            name_label = name.replace('jpg', 'png')
        else:
            name_label = name

        cur_info = '{}  {}\n'.format(os.path.join(val_image_path, name), 'val') 
        ff.writelines(cur_info)
    
    for name in test_images:
        if name.split('.')[1] == 'jpg' :
            name_label = name.replace('jpg', 'png')
        else:
            name_label = name

        cur_info = '{}  {}\n'.format(os.path.join(test_image_path, name), 'test') 
        ff.writelines(cur_info)

