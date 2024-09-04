from model import *

import tensorflow as tf
import numpy as np
import datetime
import os
import cv2 as cv

def train(model, load_train_ds, load_test_ds, train_count, test_count, EPOCH, BATCHSIZE):
    current_path = os.getcwd()

    logs_path = os.path.join(current_path, 'logs')
    checkpoints_path = os.path.join(current_path, 'checkpoints')
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    log = os.path.join(logs_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log,histogram_freq=1)
    save_func = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoints_path,'checkpoint-{epoch:02d}-{val_loss:.2f}.weights.h5'),save_weights_only=True)

    model.fit(load_train_ds, epochs=EPOCH, steps_per_epoch=train_count//BATCHSIZE, validation_data=load_test_ds,
                  validation_steps=test_count//BATCHSIZE, callbacks=[tensorboard_callback,save_func])
    model.save(os.path.join(current_path,'model.keras'))

    print("TRAIN FINISHED")

    return model