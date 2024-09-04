import eikon as ek
import yaml


# Read API key from YAML file
def read_api_key(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
        api_key = config['api_key']
    return api_key


def get_stock_data(stock, start_date, end_date, interval):
    historical_stocks = ek.get_timeseries([stock],
                                          start_date=start_date,
                                          end_date=end_date,
                                          interval=interval)
    return historical_stocks


if __name__ == "__main__":
    yaml_file = '../../config/config.yaml'

    # Read the API key from the YAML file
    api_key = read_api_key(yaml_file)

    # Set up API key
    ek.set_app_key(api_key)

    # # Set up parameters
    STOCK = "MSFT.O"
    START_DATE = "2013-01-01"
    END_DATE = "2023-12-31"
    INTERVAL = "minute"

    print(f"Get stock data from {START_DATE} to {END_DATE} ({INTERVAL})")

    STOCK_DATA = get_stock_data(STOCK, START_DATE, END_DATE, INTERVAL)

    print("Write stock data to CSV")

    STOCK_DATA.to_csv(f"../../File/stock/stock_{INTERVAL}.csv")

    # testing file
    # Set up parameters
    START_DATE = "2024-01-01"
    END_DATE = "2024-05-31"

    print(f"Get stock data from {START_DATE} to {END_DATE} ({INTERVAL})")

    STOCK_DATA = get_stock_data(STOCK, START_DATE, END_DATE, INTERVAL)

    print("Write stock data to CSV")

    STOCK_DATA.to_csv(f"../../File/stock/stock_{INTERVAL}_test.csv")