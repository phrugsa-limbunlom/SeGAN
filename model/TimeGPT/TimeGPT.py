import logging

import pandas as pd
from matplotlib import pyplot as plt
from nixtla import NixtlaClient
from sklearn.metrics import root_mean_squared_error

# h = prediction time frame
# fine tune step = number of iterations
def timegpt_forecast(df, X_df, h, step, freq, loss, level):
    return nixtla_client.forecast(
        df=df,
        X_df=X_df,
        h=h,
        level=level,
        finetune_steps=step,
        freq=freq,
        finetune_loss=loss,
        model='timegpt-1-long-horizon',
        # model='timegpt-1',
        time_col='Date',
        target_col='CLOSE',
        add_history=True
    )


def timegpt_plot(df, fcst_df):
    plot = nixtla_client.plot(df, fcst_df, models=['TimeGPT'], level=[50, 80, 90], time_col='Date', target_col='CLOSE')
    plot.savefig(f"figure/nixtla_plot/TimeGPT_{key}_{value}.png")
    plot.show()


def rmse_training_loss(df, fsct_df):
    # Filter 2024 out
    fsct_df['Date'] = pd.to_datetime(fsct_df['Date'])
    fsct_df.set_index('Date', inplace=True)

    data_history_price = fsct_df.loc[fsct_df.index.year != 2024]

    fsct_df.reset_index(inplace=True)

    filtered_date_df = pd.merge(df, data_history_price, on='Date')

    filtered_date_df.loc[:, 'TimeGPT'] = data_history_price['TimeGPT'].values

    actual_price = filtered_date_df['CLOSE'].to_numpy()

    pred_price = filtered_date_df['TimeGPT'].to_numpy()

    return root_mean_squared_error(actual_price, pred_price)


def construct_exogenous_variables(df, h):
    # List of exogenous variables
    exogenous_vars = [
        'HIGH', 'LOW', 'OPEN', 'COUNT', 'VOLUME', 'daily_sentiment_score',
        'S&P 500', 'Dow Jones', 'NASDAQ 100', 'Nikkei 225', 'FTSE 100', 'DAX 30'
    ]

    # Forecast each exogenous variable separately
    forecasted_exog_dfs = []
    for exog_var in exogenous_vars:
        logging.info(df.columns)
        df_exog = df[['Date', exog_var]].rename(columns={'Date': 'ds', exog_var: 'y'})
        timegpt_fcst_exog = nixtla_client.forecast(df=df_exog, h=h, target_col='y')
        timegpt_fcst_exog = timegpt_fcst_exog.rename(columns={'TimeGPT': exog_var})
        forecasted_exog_dfs.append(timegpt_fcst_exog[['ds', exog_var]])

    # Merge all forecasted exogenous variables
    X_df = forecasted_exog_dfs[0]
    for exog_df in forecasted_exog_dfs[1:]:
        X_df = X_df.merge(exog_df, on='ds')

    # Rename 'ds' back to 'Date'
    X_df = X_df.rename(columns={'ds': 'Date'})
    return X_df


def feature_weights(key, value):

    plt.figure(figsize=(12, 8))

    nixtla_client.weights_x.plot.barh(x='features', y='weights')

    plt.title("Feature Weights")
    plt.xlabel("Weight")
    plt.ylabel("Features")

    plt.tight_layout()

    # To save the plot as an image file
    plt.savefig(f"../TimeGPT/figure/feature_weight/feature_weights_{key}_{value}.png")

    # To display the plot
    plt.show()


if __name__ == "__main__":

    logging.basicConfig(filename='TimeGPT.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    nixtla_client = NixtlaClient(
        # defaults to os.environ.get("NIXTLA_API_KEY")
        api_key='nixtla-tok-v0dKegWAI6RynBBIIMfp7oVbGKGcKwYN26zBjtEcajRWY4h113zfKpDDnBgtLB0xgGgExxpiB65q1DoC'
    )

    # Load data
    df = pd.read_csv("../../file/dataset/train_all.csv")

    df_test = pd.read_csv("../../file/dataset/test_all.csv")

    # Ensure the date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df_test['Date'] = pd.to_datetime(df_test['Date'])

    freq_h_data = {
        'D': 152,  # Daily
        'W': 21,  # Weekly
        'ME': 5  # Monthly
    }

    level = [50, 80, 90]  # confidence levels

    for key, value in freq_h_data.items():

        # Setting the date column as the index
        df.set_index('Date', inplace=True)
        df_test.set_index('Date', inplace=True)

        if key == "D":
            logging.info(f"Prediction on {key}, {value} days ahead")
        elif key == "W":
            logging.info(f"Prediction on {key}, {value} weeks ahead")
            # Resampling the data to weekly frequency and taking the last entry of each week
            df = df.resample('W').last()
            df_test = df_test.resample('W').last()
            # Remove the extended month
            df_test = df_test.iloc[:-1]
        else:
            logging.info(f"Prediction on {key}, {value} months ahead")
            # Resampling the data to monthly frequency and taking the last entry of each month
            df = df.resample('M').last()
            df_test = df_test.resample('M').last()

        df.reset_index(inplace=True)
        df_test.reset_index(inplace=True)

        # X_df = construct_exogenous_variables(df, value)
        X_df = df_test.drop(columns=["CLOSE"])

        timegpt_forecast_finetune_df = timegpt_forecast(df=df, X_df=X_df, step=10, h=value, freq=key, loss='mse',
                                                        level=level)

        timegpt_forecast_finetune_df = timegpt_forecast_finetune_df.round(2)

        rmse = rmse_training_loss(df, timegpt_forecast_finetune_df)

        if key == "D":
            logging.info(f"RMSE training set ({key}) : {rmse:.3f}")
        elif key == "W":
            logging.info(f"RMSE training set ({key}) : {rmse:.3f}")
        else:
            logging.info(f"RMSE training set ({key}) : {rmse:.3f}")

        final_df = pd.concat([df, df_test], axis=0)
        timegpt_plot(final_df, timegpt_forecast_finetune_df)

        timegpt_forecast_finetune_df = timegpt_forecast_finetune_df.rename(columns={'TimeGPT': 'CLOSE'})

        timegpt_forecast_finetune_df.to_csv(
            f"result/timeGPT_result_{key}_{value}.csv",
            index=False)

        feature_weights(key, value)