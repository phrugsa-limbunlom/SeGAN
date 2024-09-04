import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter


def plot_graph(data, fsct, time_step, prediction_time_frame):
    data['Date'] = pd.to_datetime(data['Date'])
    fsct['Date'] = pd.to_datetime(fsct['Date'])

    data.set_index('Date', inplace=True)
    fsct.set_index('Date', inplace=True)

    if time_step == 'W':
        data = data.resample('W').last()
    elif time_step == 'ME':
        data = data.resample('M').last()

    # Filter year 2024 out
    fsct = fsct.loc[fsct.index.year != 2024]

    data.reset_index(inplace=True)
    fsct.reset_index(inplace=True)

    print(fsct.head())

    # Plot data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Date'], data["CLOSE"], label='real price', color='steelblue')
    ax.plot(fsct['Date'], fsct["CLOSE"], label='predicted price', color='hotpink')
    ax.set(xlabel="Date",
           ylabel="Closing Price (USD)",
           title="The Closing Price of MSFT stock from 2013 to 2023")
    date_form = DateFormatter("%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.legend(loc='upper right')

    # Show the plot
    plt.xticks(rotation=45)
    plt.savefig(f"figure/timeGPT_{time_step}_{prediction_time_frame}.png")
    plt.show()


if __name__ == "__main__":

    freq_h_data = {
        'D': 152,  # Daily
        'W': 21,  # Weekly
        'ME': 5  # Monthly
    }

    for key, value in freq_h_data.items():
        df = pd.read_csv("../../file/dataset/train_all.csv")

        timegpt_forecast_finetune_df = pd.read_csv(
            f"result/timeGPT_result_{key}_{value}.csv")

        print(timegpt_forecast_finetune_df.head())

        plot_graph(df, timegpt_forecast_finetune_df, key, value)
