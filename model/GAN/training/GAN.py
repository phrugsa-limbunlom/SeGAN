import tensorflow as tf
from keras import Input, Model
from keras.src.optimizers import Adam

from model.GAN.training.Discriminator import Discriminator
from model.GAN.training.Generator import Generator


class GAN:
    def __init__(self, generator_input_shape, generator_output_shape, discriminator_input_shape, learning_rate,
                 beta_1, beta_2, epsilon):
        self.generator = Generator(generator_input_shape, generator_output_shape).model
        self.discriminator = Discriminator(discriminator_input_shape).model
        # self.gan = self.build_gan(generator_input_shape)

        # Initialize optimizers here
        # self.generator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)
        # self.discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)
        self.generator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def build_gan(self, input_shape):
        self.discriminator.trainable = False
        gan_input = Input(shape=input_shape)
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        model = Model(gan_input, gan_output)
        return model

    def compile_models(self):
        self.generator.compile(optimizer=self.generator_optimizer, loss='mean_squared_error')
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss='binary_crossentropy',
                                   metrics=['accuracy'])
        # self.gan.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)

    @staticmethod
    def generator_loss(fake_output, real_price):
        # return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))
        return tf.reduce_mean(tf.keras.losses.mean_squared_error(real_price, fake_output))

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
        fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return tf.reduce_mean(total_loss)
