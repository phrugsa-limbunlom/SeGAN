import logging

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(df, price_column, risk_free_rate, EMA, RSI, trading_days=252, initial_cash=10000):
    if EMA == 2:
        # Calculate short-term and long-term EMAs
        df['EMA_short'] = df[price_column].ewm(span=10, adjust=False).mean()  # Adjusted span for short-term EMA
        df['EMA_long'] = df[price_column].ewm(span=40, adjust=False).mean()  # Adjusted span for long-term EMA
    elif EMA == 3:
        # Calculate short-term, medium-term, and long-term EMAs
        df['EMA_short'] = df[price_column].ewm(span=5, adjust=False).mean()  # 5-day EMA
        df['EMA_medium'] = df[price_column].ewm(span=15, adjust=False).mean()  # 15-day EMA
        df['EMA_long'] = df[price_column].ewm(span=30, adjust=False).mean()  # 30-day EMA

    if RSI:
        # Calculate RSI as an additional indicator
        delta = df[price_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # Set position: 1 for buy, -1 for sale, 0 for hold
    if EMA == 2 and RSI:
        df[f'position_{price_column}'] = np.where(
            (df['EMA_short'] > df['EMA_long']) & (df['RSI'] < 70), 1,
            np.where((df['EMA_short'] < df['EMA_long']) & (df['RSI'] > 30), -1,
                     0))
    elif EMA == 3 and RSI:
        df[f'position_{price_column}'] = np.where(
            (df['EMA_short'] > df['EMA_medium']) & (df['EMA_medium'] > df['EMA_long']) & (df['RSI'] < 70), 1,
            np.where((df['EMA_short'] < df['EMA_medium']) & (df['EMA_medium'] < df['EMA_long']) & (df['RSI'] > 30), -1,
                     0))

    elif EMA == 2 and not RSI:
        df[f'position_{price_column}'] = np.where(
            df['EMA_short'] > df['EMA_long'], 1,
            np.where(df['EMA_short'] < df['EMA_long'], -1, 0)
        )
    elif EMA == 3 and not RSI:
        df[f'position_{price_column}'] = np.where(
            (df['EMA_short'] > df['EMA_medium']) & (df['EMA_medium'] > df['EMA_long']), 1,
            np.where((df['EMA_short'] < df['EMA_medium']) & (df['EMA_medium'] < df['EMA_long']), -1,
                     0))

    # Calculate daily returns based on the strategy
    df[f'strategy_returns_{price_column}'] = df[f'position_{price_column}'].shift(1) * df[price_column].pct_change()

    # Calculate cumulative position to determine stock holdings
    df['shares_held'] = (df[f'position_{price_column}'].shift(1) * (initial_cash // df[price_column])).cumsum()

    # Calculate the value of the portfolio
    df['portfolio_value'] = df['shares_held'] * df[price_column]

    # Drop NaN values
    df = df.dropna()

    # Calculate excess returns
    excess_returns = df[f'strategy_returns_{price_column}'] - risk_free_rate / trading_days

    # Calculate Sharpe ratio
    sharpe_ratio = np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()

    return sharpe_ratio, df


if __name__ == '__main__':
    logging.basicConfig(filename='GAN_sharpe_ratio_comparison.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    look_back = [7, 15, 30, 60]
    time_horizons = [1]

    # Set a realistic risk-free rate (e.g., 2% annual rate)
    risk_free_rate = 0.02

    # 2 EMAs (Short, Long) / 3 EMAs (Short, Medium, Long)
    EMA = [2, 3]

    RSI = [True, False]

    for time_horizon in time_horizons:
        for step in look_back:
            logging.info(f"Running time step {step}")

            for ema in EMA:
                for rsi in RSI:
                    if ema == 2 and rsi:
                        logging.info("EMA 10, 40 with RSI")
                    elif ema == 2 and not rsi:
                        logging.info("EMA 10, 40 without RSI")
                    elif ema == 3 and rsi:
                        logging.info("EMA 5, 15, 30 with RSI")
                    elif ema == 3 and not rsi:
                        logging.info("EMA 5, 15, 30 without RSI")

                    df_real = pd.read_csv("../../../file/dataset/test_all.csv")
                    df_predicted = pd.read_csv(
                        f"../testing/prediction/daily/generated_closing_prices_{step}days_{time_horizon}days.csv")

                    # Align dataframes
                    df_aligned = pd.merge(df_real, df_predicted, on='Date', suffixes=('_REAL', '_PRED'))
                    df_aligned = df_aligned[["Date", "CLOSE", "Generated_Closing_Price"]]
                    df_aligned.columns = ['Date', 'real_price', 'predicted_price']
                    df_aligned['Date'] = pd.to_datetime(df_aligned['Date'])
                    df_aligned.set_index('Date', inplace=True)

                    logging.info(df_aligned.head())

                    # Check if there are any missing values
                    if df_aligned.isnull().values.any():
                        logging.info("Warning: There are missing values in the aligned dataframe.")

                    # Calculate Sharpe ratio for real prices
                    sharpe_ratio_real, df_with_returns_real = calculate_sharpe_ratio(df_aligned, 'real_price',
                                                                                     risk_free_rate, ema, rsi)
                    logging.info(f"Sharpe Ratio (Real Prices): {sharpe_ratio_real:.3f}")

                    # Calculate Sharpe ratio for predicted prices
                    sharpe_ratio_pred, df_with_returns_pred = calculate_sharpe_ratio(df_with_returns_real,
                                                                                     'predicted_price',
                                                                                     risk_free_rate, ema, rsi)
                    logging.info(f"Sharpe Ratio (Predicted Prices): {sharpe_ratio_pred:.3f}")
