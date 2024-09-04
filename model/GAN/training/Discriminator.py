from keras import Input
from keras.layers import Dense, Conv1D, Flatten
from keras.models import Sequential
from keras.src.layers import LeakyReLU, BatchNormalization


class Discriminator:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Dense(256),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            Dense(128),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            Dense(64),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
            # Dense(1, activation='linear')  # WGAN
        ])

        return model
        # cnn_net = Sequential()
        # cnn_net.add(
        #     Conv1D(32, input_shape=(31, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
        # cnn_net.add(Conv1D(64, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
        # cnn_net.add(Conv1D(128, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
        # cnn_net.add(Flatten())
        # cnn_net.add(Dense(220, use_bias=False))
        # cnn_net.add(LeakyReLU())
        # cnn_net.add(Dense(220, use_bias=False, activation='relu'))
        # cnn_net.add(Dense(1, activation='sigmoid'))
        # return cnn_net
