import tensorflow as tf
import numpy as np
import cv2 as cv

def channel_attention_block(x, inter_channel):
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    fc1 = tf.keras.layers.Dense(inter_channel // 16, activation='relu')(avg_pool)
    fc2 = tf.keras.layers.Dense(inter_channel, activation='sigmoid')(fc1)
    
    scale = tf.keras.layers.Reshape((1, 1, inter_channel))(fc2)

    scaled_x = tf.keras.layers.Multiply()([x, scale])
    
    return scaled_x

def fcn_model(SHAPE):
    conv_base = tf.keras.applications.VGG16(weights='imagenet',input_shape=SHAPE,include_top=False)
    base_output_layer_name = ['block5_conv3','block4_conv3','block3_conv3','block5_pool']
    base_output_layer = [conv_base.get_layer(l).output for l in base_output_layer_name]
    base_model = tf.keras.models.Model(inputs=conv_base.input,outputs=base_output_layer)
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=SHAPE)
    base_ou_b5_c3,base_ou_b4_c3,base_ou_b3_c3,base_ou_b5_p = base_model(inputs)

    base_ou_b5_p = tf.keras.layers.Conv2DTranspose(512,3,strides=2,padding='same',activation='relu')(base_ou_b5_p)
    base_ou_b5_p = tf.keras.layers.Conv2D(512,3,strides=1,padding='same',activation='relu')(base_ou_b5_p)
    base_ou_b5_p = channel_attention_block(base_ou_b5_p, 512)
    first_skip = tf.keras.layers.Lambda(lambda x: tf.add(x[0], x[1]))([base_ou_b5_p, base_ou_b5_c3])
    first_skip = tf.keras.layers.Conv2DTranspose(512,3,strides=2,padding='same',activation='relu')(first_skip)
    first_skip = tf.keras.layers.Conv2D(512,3,strides=1,padding='same',activation='relu')(first_skip)
    second_skip = channel_attention_block(first_skip, 512)
    second_skip = tf.keras.layers.Lambda(lambda x: tf.add(x[0], x[1]))([first_skip, base_ou_b4_c3])
    second_skip = tf.keras.layers.Conv2DTranspose(256,3,strides=2,padding='same',activation='relu')(second_skip)
    second_skip = tf.keras.layers.Conv2D(256,3,strides=1,padding='same',activation='relu')(second_skip)
    third_skip = channel_attention_block(second_skip, 256)
    third_skip = tf.keras.layers.Lambda(lambda x: tf.add(x[0], x[1]))([second_skip, base_ou_b3_c3])
    third_skip = tf.keras.layers.Conv2DTranspose(128,3,strides=2,padding='same',activation='relu')(third_skip)
    third_skip = tf.keras.layers.Conv2D(128,3,strides=1,padding='same',activation='relu')(third_skip)
    outputs = tf.keras.layers.Conv2DTranspose(18,3,strides=2,padding='same',activation='softmax')(third_skip)

    fcn_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    fcn_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    
    return fcn_model


def conv_block(inputs, filters, pool=True):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = channel_attention_block(x, filters)

    if pool == True:
        p = tf.keras.layers.MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x

def unet_model(SHAPE):
    inputs = tf.keras.layers.Input(shape=SHAPE)

    """ Encoder """
    x1, p1 = conv_block(inputs, 16, pool=True)
    x2, p2 = conv_block(p1, 32, pool=True)
    x3, p3 = conv_block(p2, 48, pool=True)
    x4, p4 = conv_block(p3, 64, pool=True)

    """ Bridge """
    b1 = conv_block(p4, 128, pool=False)

    """ Decoder """
    u1 = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = tf.keras.layers.Concatenate()([u1, x4])
    x5 = conv_block(c1, 64, pool=False)

    u2 = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = tf.keras.layers.Concatenate()([u2, x3])
    x6 = conv_block(c2, 48, pool=False)

    u3 = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = tf.keras.layers.Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False)

    u4 = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = tf.keras.layers.Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False)

    """ Output layer """
    output = tf.keras.layers.Conv2D(18, 1, padding="same", activation="softmax")(x8)

    model = tf.keras.Model(inputs = inputs, outputs = output)

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

    return model

def load_fcn_model(fname):
    def add_lambda(x):
        return tf.add(x[0], x[1])
        
    model = tf.keras.models.load_model(
    fname,
    custom_objects={
        'channel_attention_block': channel_attention_block,
        'lambda': add_lambda 
    })
    return model

def load_unet_model(fname):
    model = tf.keras.models.load_model(
    fname,
    custom_objects={
        'channel_attention_block': channel_attention_block,
        'conv_block': conv_block
    })
    return model