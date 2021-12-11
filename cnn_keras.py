import os
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers.wrappers import TimeDistributed as TD
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, Cropping3D, Conv2DTranspose
from tensorflow.python.keras.layers import GlobalAveragePooling2D

import donkeycar as dk

def adjust_input_shape(input_shape, roi_crop):
    height = input_shape[0]
    new_height = height - roi_crop[0] - roi_crop[1]
    return (new_height, input_shape[1], input_shape[2])

class KerasPilot(object):
    '''
    Base class for Keras models that will provide steering and throttle to guide a car.
    '''
    def __init__(self):
        self.model = None
        self.optimizer = "adam"

    def load(self, model_path):
        self.model = keras.models.load_model(model_path, compile=False)

    def load_weights(self, model_path, by_name=True):
        self.model.load_weights(model_path, by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        pass

    def set_optimizer(self, optimizer_type, rate, decay):
        # if optimizer_type == "adam":
        self.model.optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        # elif optimizer_type == "sgd":
        #     self.model.optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        # elif optimizer_type == "rmsprop":
        #     self.model.optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        # else:
        #     raise Exception("unknown optimizer type: %s" % optimizer_type)

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):

        """
        train_gen: generator that yields an array of images an array of

        """

        #checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path,
                                                    monitor='val_loss',
                                                    verbose=verbose,
                                                    save_best_only=True,
                                                    mode='min')

        #stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=min_delta,
                                                   patience=patience,
                                                   verbose=verbose,
                                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
                        train_gen,
                        steps_per_epoch=steps,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_gen,
                        callbacks=callbacks_list,
                        validation_steps=steps*(1.0 - train_split))
        return hist

class KerasVGG16(KerasPilot):
    '''
    The KerasLinear pilot uses one neuron to output a continous value via the
    Keras Dense layer with linear activation. One each for steering and throttle.
    The output is not bounded.
    '''
    def __init__(self, num_outputs=2, input_shape=(120, 160, 3), roi_crop=(0, 0), last_activation="linear", *args, **kwargs):
        super(KerasVGG16, self).__init__(*args, **kwargs)
        drop = 0.2
        self.seq_length = 5
        self.image_d = input_shape[-1]
        self.image_w = input_shape[1]
        self.image_h = input_shape[0]
        self.img_seq = []
        input_shape = adjust_input_shape(input_shape, roi_crop)
        img_seq_shape = (self.seq_length,) + input_shape
        img_in = Input(batch_shape = img_seq_shape, name='img_in')
        # img_in = Input(shape=input_shape, name='img_in')
        x = Sequential()
        self.image_processor = tf.keras.applications.VGG16(
            include_top=False,
            weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None)

        x.add(self.image_processor)
        x.add(TD(GlobalAveragePooling2D()))
        x.add(LSTM(128, return_sequences=True, name="LSTM_seq") )
        x.add(TD(Dropout(.1)))
        x.add(LSTM(128, return_sequences=False, name="LSTM_fin"))
        # x = Flatten(name='flattened')(x)
        x.add(Dense(100, activation='relu'))
        x.add(Dropout(drop))
        x.add(Dense(50, activation='relu'))
        x.add(Dropout(drop))

        # outputs = []
        # for i in range(num_outputs):
        x.add(Dense(num_outputs, activation='tanh', name='n_outputs'))
        self.model = x
        # self.model = Model(inputs=[img_in], outputs=outputs)
        # self.model = default_n_linear(num_outputs, roi_crop, last_activation=last_activation)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                loss='mse')

    def run(self, img_arr):
        # if img_arr.shape[2] == 3 and self.image_d == 1:
        # img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)

        img_arr = np.array(self.img_seq).reshape(1, self.seq_length, self.image_h, self.image_w, self.image_d )
        outputs = self.model.predict([img_arr])
        steering = outputs[0][0]
        throttle = outputs[0][1]
        return steering, throttle