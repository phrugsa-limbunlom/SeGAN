import logging
from pickle import load

import matplotlib.pyplot as plt
import numpy as np

from model.BaseLine.training.DataPreprocessor import DataPreprocessor
from model.BaseLine.training.GRUModel import GRUModel
from model.BaseLine.training.LSTMModel import LSTMModel

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(filename='base_model.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    look_back = [7, 15, 30, 60]
    time_horizons = [1, 7, 30]

    folder = ""
    for time_horizon in time_horizons:
        for step in look_back:

            logging.info(f"Train on {step} days period")

            logging.info(f"Prediction {time_horizon} day/days ahead")

            if time_horizon == 1:
                folder = "daily"
            elif time_horizon == 7:
                folder = "weekly"
            elif time_horizon == 30:
                folder = "monthly"

            # Preprocess Training Data
            X_scaler = load(open('../../../preprocessing/X_scaler.pkl', 'rb'))
            y_scaler = load(open('../../../preprocessing/y_scaler.pkl', 'rb'))
            file_path = '../../../file/dataset/train_all.csv'
            data_preprocessor = DataPreprocessor(X_scaler, y_scaler, file_path, sequence_length=step,
                                                 time_horizon=time_horizon)
            x_train, y_train, train_dates = data_preprocessor.preprocess_data()

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])

            print(x_train.shape)
            print(y_train.shape)

            # GRU Model
            gru_model = GRUModel(input_shape=(x_train.shape[1], x_train.shape[2]), output_shape=time_horizon,
                                 y_scaler=data_preprocessor.y_scaler)

            # print(gru_model.output_shape)
            gru_model.compile_model()

            print(x_train.shape)
            print(y_train.shape)

            gru_history = gru_model.train_model(x_train, y_train)

            gru_rmse = gru_model.evaluate_model(x_train, y_train)

            gru_model.model.save_weights(f'weight/{folder}/gru_model_{step}days_{time_horizon}days.weights.weights.h5')

            logging.info(f"GRU Model RMSE Training set: {gru_rmse:.3f}")

            # Plot GRU loss
            plt.figure()
            plt.plot(gru_history.history['loss'], label='Train Loss')
            plt.plot(gru_history.history['val_loss'], label='Validation Loss')
            plt.title('GRU Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'figure/{folder}/gru_loss_{step}days_{time_horizon}days.png')
            plt.show()

            # LSTM Model
            lstm_model = LSTMModel(input_shape=(x_train.shape[1], x_train.shape[2]), output_shape=time_horizon,
                                   y_scaler=data_preprocessor.y_scaler)
            lstm_model.compile_model()
            lstm_history = lstm_model.train_model(x_train, y_train)
            lstm_rmse = lstm_model.evaluate_model(x_train, y_train)

            lstm_model.model.save_weights(
                f'weight/{folder}/lstm_model_{step}days_{time_horizon}days_weights.weights.h5')

            logging.info(f"LSTM Model RMSE Training set: {lstm_rmse:.3f}")

            # Plot LSTM loss
            plt.figure()
            plt.plot(lstm_history.history['loss'], label='Train Loss')
            plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
            plt.title('LSTM Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'figure/{folder}/lstm_loss_{step}days_{time_horizon}days.png')
            plt.show()
