from utils import *

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
import cv2 as cv



def select_data(path, datalength, test_ratio):
    dirs = glob.glob(os.path.join(path,'*'))
    images = []
    labels = []
    for dir in dirs:
        images += glob.glob(os.path.join(dir, 'image/*.png'))
        labels += glob.glob(os.path.join(dir, 'indexLabel/*.png'))
    train_images, test_images, train_labels, test_labels = train_test_split(images[:datalength], labels[:datalength], test_size=test_ratio, random_state=42)
    train_count = len(train_images)
    test_count = len(test_images)
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds =  tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    return train_ds, test_ds, train_count, test_count, test_images, test_labels

@tf.function
def load(image_path, label_path):
    image = read_normal_image(image_path)
    label = read_normal_label(label_path)
    return image, label

def load_dataset(train_ds, test_ds, buffersize, BATCHSIZE):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    load_train_ds = train_ds.map(load, num_parallel_calls=AUTOTUNE)
    load_test_ds = test_ds.map(load, num_parallel_calls=AUTOTUNE)
    load_train_ds = load_train_ds.repeat().shuffle(buffersize).batch(BATCHSIZE)
    load_test_ds = load_test_ds.repeat().batch(BATCHSIZE)
    return load_train_ds, load_test_ds