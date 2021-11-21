import os
import numpy as np
import tensorflow as tf

class uNet:

    def __init__(self, imageHeight, imageWidth, in_channels=1, out_channels=1):
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.in_channels = in_channels
        self.out_channels = out_channels

    def build(self, n_levels, initial_feature=32, n_blocks=2, kernel_size=3, pooling_size=2):
        inputs = tf.keras.layers.Input(shape=(self.imageHeight, self.imageWidth, self.in_channels))
        x = inputs

        convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')

        #downstream
        skips ={}
        for level in range(n_levels):
            for _ in range(n_blocks):
                x = tf.keras.layers.Conv2D(initial_feature * (2 ** level), **convpars)(x)
            if level < n_levels - 1:
                skips[level] = x
                x = tf.keras.layers.MaxPool2D(pooling_size)(x)

        #upstream
        for level in reversed(range(n_levels-1)):
            x = tf.keras.layers.Conv2DTranspose(initial_feature * (2 ** level), strides=pooling_size, **convpars)(x)
            x = tf.keras.layers.Concatenate()([x, skips[level]])
            for _ in range(n_blocks):
                x = tf.keras.layers.Conv2D(initial_feature * (2 ** level), **convpars)(x)

        #output
        x = tf.keras.layers.Conv2D(self.out_channels, kernel_size=1, activation='sigmoid', padding='same')(x)

        return tf.keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_feature}')


if __name__ == '__main__':
    unet = uNet(320, 320)
    model = unet.build(4)
    model.summary()


