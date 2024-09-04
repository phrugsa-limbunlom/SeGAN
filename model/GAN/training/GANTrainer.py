import logging

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from model.GAN.training.EarlyStopping import EarlyStopping


class GANTrainer:
    def __init__(self, gan, epochs, look_back, time_horizon, y_scaler, batch_size=32, patience=5):
        self.gan = gan
        self.epochs = epochs
        self.look_back = look_back
        self.time_horizon = time_horizon
        self.y_scaler = y_scaler
        self.batch_size = batch_size
        self.patience = patience
        self.early_stopping = EarlyStopping(patience=self.patience)
        self.G_losses = []
        self.D_losses = []
        self.real_price = []
        self.generated_price = []
        self.train_hist = {'D_losses': [], 'G_losses': []}

    @tf.function
    def train_step(self, real_data, real_price, valid_price):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated_price = self.gan.generator(real_data, training=True)
            generated_price_reshape = tf.reshape(generated_price,
                                                 [generated_price.shape[0], generated_price.shape[1], 1])
            real_price_reshape = tf.reshape(real_price, [real_price.shape[0], real_price.shape[1], 1])

            real_output = self.gan.discriminator(tf.concat([real_price_reshape, valid_price], axis=1), training=True)
            fake_output = self.gan.discriminator(
                tf.concat([tf.cast(generated_price_reshape, tf.float64), np.random.random(valid_price.shape)], axis=1),
                training=True)

            # real_output = self.gan.discriminator(
            #     tf.concat([real_price_reshape, np.random.random(valid_price.shape)], axis=1), training=True)
            # fake_output = self.gan.discriminator(
            #     tf.concat([tf.cast(generated_price_reshape, tf.float64), valid_price], axis=1),
            #     training=True)

            d_loss = self.gan.discriminator_loss(real_output, fake_output)
            g_loss = self.gan.generator_loss(generated_price, real_price)

        gradients_of_discriminator = d_tape.gradient(d_loss, self.gan.discriminator.trainable_variables)
        gradients_of_generator = g_tape.gradient(g_loss, self.gan.generator.trainable_variables)

        self.gan.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.gan.discriminator.trainable_variables))
        self.gan.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.gan.generator.trainable_variables))

        return d_loss, g_loss, generated_price

    def train(self, X, next_y, y, folder):
        # self.gan.compile_models()

        for epoch in range(1, self.epochs + 1):
            # valid_y = np.ones((X.shape[0], 1), dtype=np.float32)

            d_loss, g_loss, generated_data = self.train_step(X, next_y, y)

            # Append scalar values, not lists
            self.train_hist['D_losses'].append(d_loss.numpy())
            self.train_hist['G_losses'].append(g_loss.numpy())

            self.real_price.append(next_y)
            self.generated_price.append(generated_data.numpy())

            if epoch % 10 == 0:
                logging.info(
                    f"Epoch {epoch}, D Loss: {d_loss.numpy():.3f}, G Loss: {g_loss.numpy():.3f}")

        self.gan.generator.save_weights(
            f'weight/{folder}/generator_weights_{self.look_back}days_{self.time_horizon}days.weights.h5')

        self.gan.discriminator.save_weights(
            f'weight/{folder}/discriminator_weights_{self.look_back}days_{self.time_horizon}days.weights.h5')

    def loss_graph(self, folder):
        epochs = np.arange(1, self.epochs + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_hist['G_losses'], label='Generator Loss', color='blue')
        plt.plot(epochs, self.train_hist['D_losses'], label='Discriminator Loss', color='red')
        plt.title('GAN Training Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'figure/{folder}/gan_losses_{self.look_back}days_{self.time_horizon}days.png')
        plt.show()
