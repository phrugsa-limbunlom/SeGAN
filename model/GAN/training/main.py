import logging
from pickle import load

import pandas as pd
from sklearn.metrics import root_mean_squared_error

from model.GAN.training.DataPreprocessor import DataPreprocessor
from model.GAN.training.GAN import GAN
from model.GAN.training.GANTrainer import GANTrainer
from model.GAN.training.Generator import Generator

if __name__ == "__main__":

    logging.basicConfig(filename='GAN.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    data = pd.read_csv("../../../file/dataset/train_all.csv")

    logging.info(data.isnull().sum())

    data['Date'] = pd.to_datetime(data['Date'])
    # data.set_index('Date', inplace=True)

    logging.info(data.columns)
    logging.info(data.shape)

    look_back = [7, 15, 30, 60]

    time_horizons = [1]

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

            X_scaler = load(open('../../../preprocessing/X_scaler.pkl', 'rb'))

            y_scaler = load(open('../../../preprocessing/y_scaler.pkl', 'rb'))

            X, y = DataPreprocessor.transform(data, X_scaler, y_scaler)

            X, next_y, y, train_dates = DataPreprocessor.create_sequences(data, step, time_horizon, X.values, y.values)

            logging.info(f"Shape of X: {X.shape}")
            logging.info(f"Shape of next y: {next_y.shape}")
            logging.info(f"Shape of y: {y.shape}")

            learning_rate = 0.001
            # learning_rate = 0.00016
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 1e-7
            epoch = 180

            logging.info(
                f"Parameters : learning rate = {learning_rate}, beta 1 = {beta_1}, beta 2 = {beta_2}, epsilon = {epsilon}")

            logging.info(f"Epochs : {epoch}")

            gan = GAN(generator_input_shape=(step, X.shape[2]),
                      generator_output_shape=time_horizon,
                      discriminator_input_shape=(step + 1, 1),
                      learning_rate=learning_rate,
                      beta_1=beta_1,
                      beta_2=beta_2,
                      epsilon=epsilon)

            gan.discriminator.trainable = True

            print("Generator model ")
            print(gan.generator.summary())

            print("Discriminator model ")
            print(gan.discriminator.summary())

            trainer = GANTrainer(gan, epochs=epoch, look_back=step, time_horizon=time_horizon, y_scaler=y_scaler)

            trainer.train(X, next_y, y, folder)

            # Plot the losses
            trainer.loss_graph(folder)

            # RMSE = trainer.rmse(step)

            # Load the trained generator model for predictions
            trained_generator = Generator(input_shape=(step, X.shape[2]), output_shape=time_horizon)

            trained_generator.model.load_weights(
                f'weight/{folder}/generator_weights_{step}days_{time_horizon}days.weights.h5')

            # Use the trained generator model to predict
            generated_closing_price = trained_generator.model.predict(X)
            generated_price = y_scaler.inverse_transform(generated_closing_price.reshape(-1, 1))

            real_price = y_scaler.inverse_transform(next_y.reshape(-1, 1))

            RMSE = root_mean_squared_error(real_price, generated_price)

            logging.info(f"GAN Train RMSE: {RMSE:.3f}")
