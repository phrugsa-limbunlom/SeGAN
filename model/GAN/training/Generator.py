from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU


class Generator:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=self.output_shape))
        return model

    # def build_model(self):
    #     model = Sequential()
    #     model.add(GRU(units=1024, return_sequences=True, input_shape=self.input_shape))
    #     model.add(Dropout(0.2))
    #     model.add(GRU(units=512, return_sequences=True))
    #     model.add(Dropout(0.2))
    #     model.add(GRU(units=256, return_sequences=False))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(128))
    #     model.add(Dense(64))
    #     model.add(Dense(units=1))
    #     return model
