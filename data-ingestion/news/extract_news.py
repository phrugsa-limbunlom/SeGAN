import os
from datetime import datetime, timedelta
import eikon as ek
import yaml
from bs4 import BeautifulSoup


# Read API key from YAML file
def read_api_key(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
        api_key = config['api_key']
    return api_key


# Get news headlines (maximum range : 100)
def get_news_headlines(stock, language, date_from, date_to, size):
    NEWS = ek.get_news_headlines(query=f'R:{stock} AND Language:{language}', date_from=date_from, date_to=date_to,
                                 count=size)
    return NEWS


# Get story from storyId
def get_news_story(news_headlines):
    story = lambda story_id: ek.get_news_story(story_id)
    news_story = news_headlines["storyId"].apply(story)
    return news_story


def extract_text_from_story(story_ids):
    extract_text = lambda news_story: BeautifulSoup(news_story, "html.parser").get_text()
    text_stories = story_ids.apply(extract_text)
    return text_stories


if __name__ == "__main__":
    yaml_file = '../../config/config.yaml'

    # Read the API key from the YAML file
    api_key = read_api_key(yaml_file)

    # Set up API key
    ek.set_app_key(api_key)

    print("Get news headlines")

    # Parameters setting
    STOCK = "MSFT.O"
    LANGUAGE = "LEN"

    # DATE_FROM = ["2023-12-01",
    #              "2023-11-01",
    #              "2023-10-01",
    #              "2023-09-01",
    #              "2023-08-01",
    #              "2023-07-01",
    #              "2023-06-01",
    #              "2023-05-01",
    #              "2023-04-01",
    #              "2023-03-01",
    #              "2023-02-01",
    #              "2023-01-01"]
    #
    # DATE_TO = ["2023-12-31",
    #            "2023-11-30",
    #            "2023-10-31",
    #            "2023-09-30",
    #            "2023-08-31",
    #            "2023-07-31",
    #            "2023-06-30",
    #            "2023-05-31",
    #            "2023-04-30",
    #            "2023-03-31",
    #            "2023-02-28",
    #            "2023-01-31"]

    DATE_FROM = ["2024-05-01",
                 "2024-04-01",
                 "2024-03-01",
                 "2024-02-01",
                 "2024-01-01"]

    DATE_TO = ["2024-05-31",
               "2024-04-30",
               "2024-03-31",
               "2024-02-28",
               "2024-01-31"]

    SIZE = 100

    for i in range(0, len(DATE_FROM)):

        print(f"Date From : {DATE_FROM[i]}")

        print(f"Date To : {DATE_TO[i]}")

        start_date = datetime.strptime(DATE_FROM[i], '%Y-%m-%d')
        end_date = datetime.strptime(DATE_TO[i], '%Y-%m-%d')

        print(f"Date from {start_date} to {end_date}")

        current_date = start_date

        while current_date <= end_date:

            print(f"Current Date : {current_date}")

            try:
                NEWS = get_news_headlines(stock=STOCK, language=LANGUAGE, date_from=current_date,
                                          date_to=current_date + timedelta(days=1), size=SIZE)
            except Exception as e:
                print("No headlines available in the specified date")

            current_date += timedelta(days=1)

            print("Write news to CSV")

            file_name = "../../file/news/news_test.csv"

            print("Final news dataframe")
            print(NEWS.head())

            # Check if the file exists and if it is empty
            if not os.path.isfile(file_name) or os.stat(file_name).st_size == 0:
                # file doesn't exist or is empty, write with header
                NEWS.to_csv(file_name, mode='w', index=False, header=True)
            else:
                # file exists and is not empty, append without header
                NEWS.to_csv(file_name, mode='a', index=False, header=False)