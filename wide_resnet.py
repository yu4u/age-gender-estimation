# This code is imported from the following project: https://github.com/titu1994/Wide-Residual-Networks

import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"

        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    # Wide residual network http://arxiv.org/abs/1605.07146
    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
            # format of conv_params:
            #               [ [nb_col="kernel width", nb_row="kernel height",
            #               subsample="(stride_vertical,stride_horizontal)",
            #               border_mode="same" or "valid"] ]
            # B(3,3): orignal <<basic>> block
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)
                    convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                          subsample=v[2],
                                          border_mode=v[3],
                                          init=self._weight_init,
                                          W_regularizer=l2(self._weight_decay),
                                          bias=self._use_bias)(convs)
                else:
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                          subsample=v[2],
                                          border_mode=v[3],
                                          init=self._weight_init,
                                          W_regularizer=l2(self._weight_decay),
                                          bias=self._use_bias)(convs)

            # Shortcut Conntection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shortcut = Convolution2D(n_output_plane, nb_col=1, nb_row=1,
                                         subsample=stride,
                                         border_mode="same",
                                         init=self._weight_init,
                                         W_regularizer=l2(self._weight_decay),
                                         bias=self._use_bias)(net)
            else:
                shortcut = net

            return merge([convs, shortcut], mode="sum")

        return f


    # "Stacking Residual Units on the same stage"
    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        conv1 = Convolution2D(nb_filter=n_stages[0], nb_row=3, nb_col=3,
                              subsample=(1, 1),
                              border_mode="same",
                              init=self._weight_init,
                              W_regularizer=l2(self._weight_decay),
                              bias=self._use_bias)(inputs)  # "One conv at the beginning (spatial size: 32x32)"

        # Add wide residual blocks
        block_fn = self._wide_basic
        conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(conv1)
        conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(conv2)
        conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(conv3)
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        # Classifier block
        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode="same")(relu)
        flatten = Flatten()(pool)
        predictions_g = Dense(output_dim=2, init=self._weight_init, bias=self._use_bias,
                              W_regularizer=l2(self._weight_decay), activation="softmax")(flatten)
        predictions_a = Dense(output_dim=101, init=self._weight_init, bias=self._use_bias,
                              W_regularizer=l2(self._weight_decay), activation="softmax")(flatten)

        model = Model(input=inputs, output=[predictions_g, predictions_a])

        return model

