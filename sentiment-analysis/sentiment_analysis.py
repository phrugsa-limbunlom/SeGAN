import pandas as pd
import eikon as ek
import yaml
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


# Read API key from YAML file
def read_api_key(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
        api_key = config['api_key']
    return api_key


# Load pre-trained FinBERT model and tokenizer
def create_model_and_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


# Initialize sentiment analysis pipeline
def create_pipeline(model, tokenizer):
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# Get story from storyId
def get_news_story(news_headlines):
    story = lambda story_id: ek.get_news_story(story_id)
    news_story = news_headlines.apply(story)
    return news_story


def extract_text_from_story(story_ids):
    extract_text = lambda news_story: BeautifulSoup(news_story, "html.parser").get_text()
    text_stories = story_ids.apply(extract_text)
    return text_stories


# Predict sentiment scores for the news data
def get_sentiment_score(text):
    finbert = pipeline
    result = finbert(text)[0]
    print(result)
    # score = result['score'] if result['label'] == 'positive' else -result['score']
    return result


if __name__ == "__main__":

    NEWS = pd.read_csv("../file/news/news_test.csv")

    print("Get sentiment score from the news stories")

    # Initialize model
    MODEL_NAME = "yiyanghkust/finbert-tone"

    model, tokenizer = create_model_and_tokenizer(MODEL_NAME)

    pipeline = create_pipeline(model, tokenizer)

    # Get sentiment score

    NEWS['sentiment'] = NEWS['text'].apply(get_sentiment_score)

    NEWS.to_csv("../../file/sentiment/sentiment_test.csv")