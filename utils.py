import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt

def all_data(path):
    dirs = glob.glob(os.path.join(path,'*'))
    images = []
    labels = []
    for dir in dirs:
        images += glob.glob(os.path.join(dir, 'image/*.png'))
        labels += glob.glob(os.path.join(dir, 'indexLabel/*.png'))
    return images, labels

def read_normal_image(file_path, resize_shape = (768,1024)):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, resize_shape, method=tf.image.ResizeMethod.BICUBIC)
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1
    return image
    
def read_normal_label(file_path, resize_shape = (768,1024)):
    label = tf.io.read_file(file_path)
    label = tf.image.decode_png(label, channels=3)
    label = tf.image.resize(label, resize_shape, method=tf.image.ResizeMethod.BICUBIC)
    label = tf.clip_by_value(label, clip_value_min=1, clip_value_max=18)
    label = tf.cast(label, dtype=tf.int32)[:,:,0]
    label = label - 1
    return label

color_map = np.array([
    [0, 0, 0],           # unlabelled 0
    [60, 180, 75],       # dirt 1
    [255, 225, 25],      # mud 2
    [0, 130, 200],       # water 3
    [145, 30, 180],      # gravel 4
    [70, 240, 240],      # other-terrain 5
    [240, 50, 230],      # tree-trunk 6
    [210, 245, 60],      # tree-foliage 7
    [250, 190, 190],     # other-object 8
    [0, 128, 128],       # fence 9
    [255, 215, 180],     # structure 10
    [255, 250, 200],     # bush 11
    [0, 0, 0],           # unlabelled 12
    [170, 255, 195],     # rock 13
    [128, 128, 0],       # log 14
    [0, 0, 0],            # unlabelled 15
    [0, 0, 128],         # sky 16
    [128, 128, 128],     # grass 17
], dtype=np.uint8)

def reduc_result(result):
    result = tf.image.resize(result, (1512,2016), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    result = tf.cast(result, dtype=tf.uint8).numpy()[:,:,0]
    
    kernel = np.ones((11,11),np.uint8)
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
    result = cv.morphologyEx(result, cv.MORPH_CLOSE, kernel)
    return result

def get_result(result, path, i, map=color_map):
    color_map_tf = tf.constant(map)
    mask = tf.gather(color_map_tf, tf.cast(result, dtype=tf.int32))
    encoded_mask = tf.image.encode_png(mask)
    tf.io.write_file(os.path.join(path,f'{i}.png'), encoded_mask)


def get_mask_result(result, image, path, i, map=color_map):
    color_map_tf = tf.constant(map)
    mask = tf.gather(color_map_tf, tf.cast(result, dtype=tf.int32))
    maskpng = cv.addWeighted(mask.numpy(), 0.5, image.numpy(), 0.5, 0)
    cv.imwrite(os.path.join(path,f'result{i}.png'), maskpng)
    
def get_miou(intersection, union):
    miou = intersection / (union + 1e-6)
    avg_miou = tf.reduce_mean(miou).numpy()
    categories = ['dirt','mud','water','gravel','other_terrain','tree-trunk','tree-foliage','other-object','fence',
                  'structure','bush','rock','log','sky','grass']
    values = [miou[c].numpy() for c in [1,2,3,4,5,6,7,8,9,10,11,13,14,16,17]]

    plt.barh(categories, values, color='skyblue')
    plt.title(f'mIoU = {avg_miou}')
    plt.savefig('mIoU.png',format='png')