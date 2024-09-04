from dataset import *
from utils import *

import os
import tensorflow as tf
import cv2 as cv

def predict(model, pre_images, pre_labels):
    current_path = os.getcwd()

    results_path = os.path.join(current_path, 'results')
    mask_results_path = os.path.join(current_path, 'mask_results')
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(mask_results_path, exist_ok=True)

    intersection = tf.zeros([18], dtype=tf.float32)
    union = tf.zeros([18], dtype=tf.float32)

    for i, (image_path, label_path) in enumerate(zip(pre_images, pre_labels)):
        back = tf.io.read_file(image_path)
        back = tf.image.decode_png(back, channels=3)

        image = read_normal_image(image_path)
        image = np.expand_dims(image, 0)

        result = model.predict(image)
        result = tf.argmax(result, axis=-1)
        result = result[...,tf.newaxis]
        result = result[0]
        result = reduc_result(result)

        label = tf.io.read_file(label_path)
        label = tf.image.decode_png(label, channels=3)
        label = tf.cast(label, dtype=tf.int32)[:,:,0] - 1

        get_result(result, results_path, i)

        get_mask_result(result, back, mask_results_path, i)

        for c in range(18):
            result_c = tf.cast(tf.equal(result, c), dtype=tf.float32)
            label_c = tf.cast(tf.equal(label, c), dtype=tf.float32)

            intersection_c = tf.reduce_sum(result_c * label_c)
            union_c = tf.reduce_sum(tf.maximum(result_c, label_c))

            intersection = tf.tensor_scatter_nd_update(intersection, [[c]], [intersection_c])
            union = tf.tensor_scatter_nd_update(union, [[c]], [union_c])

    get_miou(intersection, union)

    print("PREDICT FINISHED")



    












