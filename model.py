import keras.backend as K
import keras.regularizers
import numpy as np
from sklearn.decomposition import PCA
import keras.layers as KL
import tensorflow as tf
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
# from keras.optimizers import SGD
# from sklearn.metrics import roc_auc_score


def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name,
               kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Conv2d_BN_1(x, nb_filter, kernel_size, weights, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name,
               kernel_regularizer=keras.regularizers.l2(weights))(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Unet_a(inpt):
    # inpt = Input(shape=(256, 256, 3))

    conv1 = Conv2d_BN(inpt, 8, (3, 3))
    conv1 = Conv2d_BN(conv1, 8, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 16, (3, 3))
    conv2 = Conv2d_BN(conv2, 16, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 32, (3, 3))
    conv3 = Conv2d_BN(conv3, 32, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 64, (3, 3))
    conv4 = Conv2d_BN(conv4, 64, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 128, (3, 3))
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2d_BN(conv5, 128, (3, 3))
    conv5 = Dropout(0.3)(conv5)

    out1 = KL.GlobalAveragePooling2D()(conv5)

    return out1


class MODEL:
    def __init__(self):
        self.inpt1 = Input(shape=(300, 256, 1))
        self.inpt2 = Input(shape=(300, 256, 1))
        self.inpt3 = Input(shape=832)
        self.inpt4 = Input(shape=10)
        self.inpt5 = Input(shape=1)

    def _model(self):
        x = Unet_a(self.inpt1)
        y = Unet_a(self.inpt2)
        z = Dense(256)(self.inpt3)
        z = Dense(64)(z)
        x = Concatenate()([x, z])
        x = Dense(32)(x)
        y = Concatenate()([y, z])
        y = Dense(32)(y)
        x = Concatenate()([x, y])
        x = Dense(16)(x)
        x = Concatenate()([x, self.inpt4])
        x = Dense(2)(x)
        x = Softmax(name='out')(x)
        model = Model([self.inpt1, self.inpt2, self.inpt3, self.inpt4], x)
        return model

    def model_add_Attention(self):
        x = Unet_a(self.inpt1)
        y = Unet_a(self.inpt2)

        Features = Conv1D(32, 1, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(1e-3))(self.inpt3[..., None])
        Features = Dropout(.1)(Features)
        Features = BatchNormalization()(Features)
        Features = AveragePooling1D(strides=4)(Features)

        Features = Conv1D(64, 1, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(1e-3))(Features)
        Features = Dropout(.1)(Features)
        Features = BatchNormalization()(Features)
        Features = AveragePooling1D(strides=4)(Features)

        Features = Conv1D(128, 1, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(1e-3))(Features)
        Features = Dropout(.1)(Features)
        Features = BatchNormalization()(Features)
        Features = GlobalAveragePooling1D()(Features)

        x = Concatenate()([x, y])
        x = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)

        key = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        query = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        value = x[..., None]
        key = Multiply()([key, Features])[..., None]
        query = Multiply()([query, Features])[:, None]
        x = tf.matmul(key, query)
        x = tf.matmul(x, value)[..., 0]
        x = Dropout(.1)(x)
        x = Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        x = Concatenate()([x, self.inpt4])
        x = Dropout(.1)(x)
        x = Dense(2, activation='softmax', kernel_regularizer=keras.regularizers.l2(1e-3))(x)

        return Model([self.inpt1, self.inpt2, self.inpt3, self.inpt4], x)

    def model_add_Attention_deep_learning_TNM(self):
        x = Unet_a(self.inpt1)
        y = Unet_a(self.inpt2)

        Features = Conv1D(32, 1, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(1e-3))(self.inpt3[..., None])
        Features = BatchNormalization()(Features)
        Features = AveragePooling1D(strides=4)(Features)

        Features = Conv1D(64, 1, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(1e-3))(Features)
        Features = BatchNormalization()(Features)
        Features = AveragePooling1D(strides=4)(Features)

        Features = Conv1D(128, 1, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(1e-3))(Features)
        Features = BatchNormalization()(Features)
        Features = GlobalAveragePooling1D()(Features)


        x = Concatenate()([x, y])
        x = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)

        key = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        query = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        value = x[..., None]
        key = Multiply()([key, Features])[..., None]
        query = Multiply()([query, Features])[:, None]
        x = tf.matmul(key, query)
        x = tf.matmul(x, value)[..., 0]
        x = Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        x = Concatenate()([x, self.inpt5])
        x = Dense(2, activation='softmax', kernel_regularizer=keras.regularizers.l2(1e-3))(x)

        return Model([self.inpt1, self.inpt2, self.inpt3, self.inpt5], x)

    def model_add_Attention_deep_learoning_clincial(self):
        x = Unet_a(self.inpt1)
        y = Unet_a(self.inpt2)

        x = Concatenate()([x, y])
        x = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        x = Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        x = Concatenate()([x, self.inpt4])
        x = Dense(2, activation='softmax', kernel_regularizer=keras.regularizers.l2(1e-3))(x)

        return Model([self.inpt1, self.inpt2, self.inpt4], x)

    def model_add_Attention_deep_learning_radiomocs(self):
        x = Unet_a(self.inpt1)
        y = Unet_a(self.inpt2)

        Features = Conv1D(32, 1, activation='relu')(self.inpt3[..., None])
        Features = BatchNormalization()(Features)
        Features = AveragePooling1D(strides=4)(Features)

        Features = Conv1D(64, 1, activation='relu')(Features)
        Features = BatchNormalization()(Features)
        Features = AveragePooling1D(strides=4)(Features)

        Features = Conv1D(128, 1, activation='relu')(Features)
        Features = BatchNormalization()(Features)
        Features = GlobalAveragePooling1D()(Features)


        x = Concatenate()([x, y])
        x = Dense(128, activation='relu')(x)

        key = Dense(128, activation='relu')(x)
        query = Dense(128, activation='relu')(x)
        value = x[..., None]
        key = Multiply()([key, Features])[..., None]
        query = Multiply()([query, Features])[:, None]
        x = tf.matmul(key, query)
        x = tf.matmul(x, value)[..., 0]
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='softmax')(x)

        return Model([self.inpt1, self.inpt2, self.inpt3], x)


    def model_add_Attention_onlydeepfeature(self):
        x = Unet_a(self.inpt1)
        y = Unet_a(self.inpt2)
        x = Concatenate()([x, y])
        x = Dense(128, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(2, activation='softmax')(x)
        return Model([self.inpt1, self.inpt2], x)


if __name__ == '__main__':
    MODEL().model_add_Attention().summary()