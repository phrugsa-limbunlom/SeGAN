from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from model.BaseLine.training.BaseModel import BaseModel

class LSTMModel(BaseModel):
    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=self.output_shape))
        return model
