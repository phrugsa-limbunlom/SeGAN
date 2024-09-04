import logging
from pickle import load

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
from sklearn.metrics import root_mean_squared_error

from model.GAN.training.Generator import Generator
from model.GAN.training.main import DataPreprocessor

if __name__ == "__main__":
    logging.basicConfig(filename='GAN_test.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    df_test = pd.read_csv('../../../file/dataset/test_all.csv')

    X_scaler = load(open('../../../preprocessing/X_scaler_test.pkl', 'rb'))

    y_scaler = load(open('../../../preprocessing/y_scaler_test.pkl', 'rb'))

    look_back = [7, 15, 30, 60]

    time_horizons = [1]

    folder = ""

    for time_horizon in time_horizons:
        for step in look_back:

            logging.info(f"Test on {step} days period")

            logging.info(f"Prediction {time_horizon} day/days ahead")

            if time_horizon == 1:
                folder = "daily"
            elif time_horizon == 7:
                folder = "weekly"
            elif time_horizon == 30:
                folder = "monthly"

            X, y = DataPreprocessor.transform(df_test, X_scaler, y_scaler)

            X, next_y, y, test_dates = DataPreprocessor.create_sequences(df_test, step, time_horizon, X.values,
                                                                         y.values)

            # Load the trained generator model for predictions
            trained_generator = Generator(input_shape=(step, X.shape[2]), output_shape=time_horizon)

            trained_generator.model.load_weights(
                f'../training/weight/{folder}/generator_weights_{step}days_{time_horizon}days.weights.h5')

            # Use the trained generator model to predict
            generated_closing_price = trained_generator.model.predict(X)

            real_closing_price = y_scaler.inverse_transform(next_y.reshape(-1, 1))
            generated_closing_price = y_scaler.inverse_transform(generated_closing_price)

            generated_closing_price = generated_closing_price.round(2)

            df = pd.DataFrame({
                'Date': test_dates,
                'Real_Closing_Price': real_closing_price.flatten(),
                'Generated_Closing_Price': generated_closing_price.flatten()
            })

            # Write the DataFrame to a CSV file
            filename = f'prediction/{folder}/generated_closing_prices_{step}days_{time_horizon}days.csv'
            df.to_csv(filename, index=False)

            test_dates = pd.to_datetime(test_dates)

            rmse = root_mean_squared_error(real_closing_price, generated_closing_price)
            logging.info(f'RMSE: {rmse:.3f}')

            df['Date'] = pd.to_datetime(df['Date'])

            # Set 'Date' as the index
            df.set_index('Date', inplace=True)

            # Resample data for the price change over time period
            resample_df = df.resample(f'{step}D').first().reset_index()

            df = df.reset_index()

            # Compute price changes
            df[f'Real_{step}Day_Change'] = df['Real_Closing_Price'].diff(periods=step)
            df[f'Generated_{step}Day_Change'] = df['Generated_Closing_Price'].diff(periods=step)

            # Drop the first rows with NaN values
            df = df.dropna()

            # Plot data
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.plot(df['Date'], df['Real_Closing_Price'], label='Real Closing Prices')
            ax.plot(df['Date'], df['Generated_Closing_Price'], label='Predicted Closing Prices')
            ax.set(xlabel='Date',
                   ylabel='Closing Price (USD)',
                   title=f'Real vs Predicted Closing Prices (GAN)')
            date_form = DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            ax.legend(loc='upper right')

            plt.xticks(rotation=45)

            # Show the plot
            plt.savefig(f"figure/{folder}/GAN_{step}days_{time_horizon}days.png")
            plt.show()

            # Plot data for resampling
            fig1, ax = plt.subplots(figsize=(14, 10))
            ax.plot(resample_df['Date'], resample_df['Real_Closing_Price'], label='Real Closing Prices')
            ax.plot(resample_df['Date'], resample_df['Generated_Closing_Price'], label='Predicted Closing Prices')
            ax.set(xlabel='Date',
                   ylabel='Closing Price (USD)',
                   title=f'Real vs Predicted Closing Prices (GAN) over {step} days')
            date_form = DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            ax.legend(loc='upper right')

            plt.xticks(rotation=45)

            # Show the plot
            plt.savefig(f"figure/{folder}/GAN_resample_{step}days_{time_horizon}days.png")
            plt.show()

            # Plot data for price changes

            # Plot the results
            plt.figure(figsize=(14, 7))
            plt.plot(df['Date'], df[f'Real_{step}Day_Change'], label=f'Actual {step}-day Change', color='blue')
            plt.plot(df['Date'], df[f'Generated_{step}Day_Change'], label=f'Predicted {step}-day Change', color='orange')

            plt.title(f'Real vs Predicted {step}-day Closing Price Changes (GAN)')
            plt.xlabel('Date')
            plt.ylabel(f'{step}-day Price Change (USD)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.savefig(f"figure/{folder}/GAN_price_changes_{step}days_{time_horizon}days.png")
            plt.show()