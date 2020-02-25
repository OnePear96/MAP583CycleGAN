import tensorflow as tf
from basic_models.blocks import upsample,downsample

OUTPUT_CHANNELS = 3


def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)






'''

class generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.down_stack = [
            downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
            downsample(128, 4), # (bs, 64, 64, 128)
            downsample(256, 4), # (bs, 32, 32, 256)
            downsample(512, 4), # (bs, 16, 16, 512)
            downsample(512, 4), # (bs, 8, 8, 512)
            downsample(512, 4), # (bs, 4, 4, 512)
            downsample(512, 4), # (bs, 2, 2, 512)
            downsample(512, 4), # (bs, 1, 1, 512)
        ]
        self.up_stack = [
            upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            upsample(512, 4), # (bs, 16, 16, 1024)
            upsample(256, 4), # (bs, 32, 32, 512)
            upsample(128, 4), # (bs, 64, 64, 256)
            upsample(64, 4), # (bs, 128, 128, 128)
        ]
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')

    def call(self,inputs):
        skips = []
        x = inputs
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        
        x = self.last(x)

        return x


'''
