import logging

import numpy as np
import pandas as pd


def calculate_forecast_accuracy(df, horizon):
    """
    Calculate forecasted price changes and compare with real price changes for a given horizon.
    Provides the ratio of correct and incorrect forecasts.

    :param df: DataFrame with Date, Generated_Closing_Price, and Real_Closing_Price
    :param horizon: Day horizon to forecast
    :return: DataFrame with forecasted changes, real changes, and accuracy for the horizon,
             along with the ratio of correct to incorrect forecasts
    """
    changes = pd.DataFrame(index=df.index)

    # Calculate forecasted changes
    changes[f'forecast_change_{horizon}d'] = df['Generated_Closing_Price'].shift(-horizon) - df[
        'Generated_Closing_Price']

    # Calculate real changes
    changes[f'real_change_{horizon}d'] = df['Real_Closing_Price'].shift(-horizon) - df['Real_Closing_Price']

    # Calculate accuracy (1 if both changes have the same sign or both are zero, 0 otherwise)
    changes[f'accuracy_{horizon}d'] = (
            (changes[f'forecast_change_{horizon}d'] > 0) & (changes[f'real_change_{horizon}d'] > 0) |
            (changes[f'forecast_change_{horizon}d'] < 0) & (changes[f'real_change_{horizon}d'] < 0) |
            (changes[f'forecast_change_{horizon}d'] == 0) & (changes[f'real_change_{horizon}d'] == 0)
    ).astype(int)

    # Count correct and incorrect forecasts
    correct_forecasts = changes[f'accuracy_{horizon}d'].sum()
    total_forecasts = changes[f'accuracy_{horizon}d'].count()
    incorrect_forecasts = total_forecasts - correct_forecasts

    correct_forecasts_ratio = correct_forecasts / total_forecasts

    incorrect_forecasts_ratio = incorrect_forecasts / total_forecasts

    # Calculate overall accuracy percentage
    accuracy_percentage = (correct_forecasts / total_forecasts) * 100

    return accuracy_percentage, correct_forecasts_ratio, incorrect_forecasts_ratio


if __name__ == "__main__":

    logging.basicConfig(filename='GAN_accuracy_test.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    look_back = [7, 15, 30, 60]

    for step in look_back:
        # Read the CSV file
        df = pd.read_csv(
            f"prediction/daily/generated_closing_prices_{step}days_1days.csv")

        # Sort the dataframe by date to ensure chronological order
        df = df.sort_values('Date')

        logging.info(f"Sequence length: {step}")

        # Calculate forecast accuracy
        accuracy, correct_forecast, incorrect_forecast = calculate_forecast_accuracy(df, step)
        logging.info(f"Correct forecast ratio: {correct_forecast:.3f}")
        logging.info(f"Incorrect forecast ratio: {incorrect_forecast:.3f}")
        logging.info(f"Accuracy: {accuracy:.3f}")
