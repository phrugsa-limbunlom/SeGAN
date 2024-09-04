import logging

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
from sklearn.metrics import root_mean_squared_error


def plot_graph(real, predict):
    # Plot data
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(real['Date'], real["CLOSE"], label='real price', color='steelblue')
    ax.plot(predict['Date'], predict["CLOSE"], label='predicted price',
            color='hotpink')
    ax.set(xlabel="Date",
           ylabel="Closing Price (USD)",
           title=f"The Closing Price of MSFT stock 2024")
    date_form = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    ax.legend(loc='upper right')

    plt.xticks(rotation=45)

    # Show the plot
    plt.savefig(f"figure/test/TimeGPT_{key}_{value}.png")
    plt.show()


if __name__ == "__main__":

    logging.basicConfig(filename='TimeGPT_test.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    data_real_price = pd.read_csv('../../file/dataset/test_all.csv')

    freq_h_data = {
        'D': 152,  # Daily
        'W': 21,  # Weekly
        'ME': 5  # Monthly
    }

    for key, value in freq_h_data.items():
        file_name = f"result/timeGPT_result_{key}_{value}.csv"

        data_pred_price = pd.read_csv(file_name)

        logging.info(f"File name : {file_name}")

        # Ensure the date column is of datetime type
        data_pred_price['Date'] = pd.to_datetime(data_pred_price['Date'])
        data_real_price['Date'] = pd.to_datetime(data_real_price['Date'])

        # Setting the date column as the index
        data_real_price.set_index('Date', inplace=True)
        data_pred_price.set_index('Date', inplace=True)

        if key == "W":
            # Resampling the data to weekly frequency and taking the last entry of each week
            data_real_price = data_real_price.resample('W').last()
            # Remove the extended month
            data_real_price = data_real_price.iloc[:-1]

        elif key == "ME":
            # Resampling the data to weekly frequency and taking the last entry of each week
            data_real_price = data_real_price.resample('M').last()

        print(data_real_price.index.unique())

        # Filter predict price only 2024
        data_pred_price = data_pred_price.loc[data_pred_price.index.year == 2024]
        print(data_pred_price.index.unique())

        # Resetting the index to make 'date' a column again
        data_real_price.reset_index(inplace=True)
        data_pred_price.reset_index(inplace=True)

        rmse = root_mean_squared_error(data_real_price['CLOSE'].to_numpy(), data_pred_price['CLOSE'].to_numpy())

        if key == "D":
            logging.info(f"RMSE ({key}) Prediction on {value} days: {rmse:.3f}")
        elif key == "W":
            logging.info(f"RMSE ({key}) Prediction on {value} weeks: {rmse:.3f}")
        else:
            logging.info(f"RMSE ({key}) Prediction on {value} months: {rmse:.3f}")

        plot_graph(data_real_price, data_pred_price)

        # Combine the DataFrames on the DATE column
        combined_df = pd.merge(data_real_price, data_pred_price, on='Date', suffixes=('_REAL', '_PRED'))

        # Extract the required columns
        combined_df = combined_df[['Date', 'CLOSE_REAL', 'CLOSE_PRED']]

        combined_df.to_csv(f"test/real_and_predicted_price_{key}_{value}.csv", index=False)
