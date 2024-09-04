import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def naive(file_path):
    data = pd.read_csv(file_path)
    prices = data['CLOSE'].values.reshape(-1, 1)

    # "Same Price as Yesterday" Model (Baseline)
    # Use prices[:-1] as the predictions for the next day's prices
    predicted_prices = prices[1:]
    actual_prices = prices[:-1]

    return predicted_prices, actual_prices


def plot_predictions(test, predictions, model_name):
    plt.figure(figsize=(14, 7))
    plt.plot(test, label='Actual Closing Prices', color='blue')
    plt.plot(predictions, label=f'Predicted Closing Prices ({model_name})', color='orange')

    plt.title(f'Actual vs Predicted Closing Prices ({model_name})')
    plt.xlabel('Days')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}.png")
    plt.show()


if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(filename='baseline_model.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # File paths
    train_file_path = '../../file/dataset/train_all.csv'
    test_file_path = '../../file/dataset/test_all.csv'

    # Naive Baseline RMSE (Training)
    actual_prices, predicted_prices = naive(train_file_path)

    rmse_baseline_train = root_mean_squared_error(actual_prices, predicted_prices)

    logging.info(f"Naive Model RMSE Training set: {rmse_baseline_train:.3f}")

    # Naive Baseline RMSE (Testing)
    actual_prices, predicted_prices = naive(test_file_path)

    rmse_baseline_test = root_mean_squared_error(actual_prices, predicted_prices)

    logging.info(f"Naive Model RMSE Testing set: {rmse_baseline_test:.3f}")

    plot_predictions(actual_prices, predicted_prices, "Naive")