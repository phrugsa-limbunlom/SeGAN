import logging

import numpy as np
import pandas as pd


def calculate_forecast_changes(df, horizon):
    """
    Calculate forecasted price changes for given horizons.

    :param df: DataFrame with Date and Generated_Closing_Price
    :param horizons: Day horizons to forecast
    :return: DataFrame with forecasted changes for each horizon
    """

    changes = pd.DataFrame(index=df.index)

    changes[f'change_{horizon}d'] = df['Generated_Closing_Price'].shift(-horizon) - df['Generated_Closing_Price']

    return changes


def trading_system(df, forecast_changes, horizon, risk_free_rate, trading_days):
    """
    Simulate a trading system based on GAN forecasts.

    :param df: DataFrame with Date and Real_Closing_Price
    :param forecast_changes: Series with forecasted 7-day changes
    :param risk_free_rate: Annual risk-free rate (default 2%)
    :return: Sharpe ratio, profits
    """

    holdings = 0  # 0 means no stock, 1 means holding stock
    profits = [0]  # Start with 0 profit

    for i in range(len(df) - 1):  #
        forecast_change = forecast_changes.iloc[i]

        if forecast_change[f'change_{horizon}d'] > 0:
            if holdings == 0:
                # Buy
                holdings = 1
                profits.append(profits[-1] - df.iloc[i]['Generated_Closing_Price'])
            else:
                # Hold
                profits.append(profits[-1])
        else:
            if holdings == 1:
                # Sell
                holdings = 0
                profits.append(profits[-1] + df.iloc[i]['Generated_Closing_Price'])
            else:
                # Stay out of the market
                profits.append(profits[-1])

    # If still holding at the end, sell at the last price
    if holdings == 1:
        profits[-1] += df.iloc[-1]['Generated_Closing_Price']

    # Calculate daily returns
    daily_returns = np.diff(profits) / np.abs(profits[:-1])
    daily_returns = daily_returns[np.isfinite(daily_returns)]  # Remove any infinity values

    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate, trading_days)

    return sharpe_ratio, profits


def calculate_sharpe_ratio(returns, risk_free_rate, trading_days):
    """
    Calculate the Sharpe ratio given an array of returns.

    :param returns: Array of daily returns
    :param risk_free_rate: Annual risk-free rate
    :return: Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / trading_days)  # Assuming 252 trading days in a year
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


if __name__ == "__main__":

    logging.basicConfig(filename='GAN_sharpe_ratio_comparison2.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    look_back = [7, 15, 30, 60]

    risk_free_rate = 0.02
    trading_days = 252

    for step in look_back:
        # Read the CSV file
        df = pd.read_csv(
            f"../testing/prediction/daily/generated_closing_prices_{step}days_1days.csv")

        # Sort the dataframe by date to ensure chronological order
        df = df.sort_values('Date')

        # Calculate forecast changes
        forecast_changes = calculate_forecast_changes(df, step)

        # Run the trading system
        sharpe_ratio, profit_curve = trading_system(df, forecast_changes, step, risk_free_rate, trading_days)

        logging.info(f"Sequence length: {step}")

        logging.info(f"Risk free rate: {risk_free_rate:.2f}")

        logging.info(f"Trading days: {trading_days}")

        logging.info(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        logging.info(f"Final Profit: ${profit_curve[-1]:.2f}")
