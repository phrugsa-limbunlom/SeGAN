<h1 align="center"> SE-GAN: Sentiment-Enhanced GAN for Stock Price Forecasting - A Comprehensive Analysis of Short-Term Prediction </h1>
<br>

<p align="center">
SE-GAN: Sentiment-Enhanced GAN for Stock Price Forecasting
  <br>
  <a href="https://github.com/phrugsa-limbunlom/SE-GAN-FOR-STOCK-FORECASTING/blob/main/CE901_Dissertation_2311569.pdf">(Dissertation)</a>
  <a href="https://github.com/phrugsa-limbunlom/SE-GAN-FOR-STOCK-FORECASTING/blob/main/Dissertation_PDO_2311569.pptx">(PDO)</a>
  <br>
  <br>
  Author: Phrugsa Limbunlom (CSEE, University of Essex, 2024)
</p>
<br>

<div align="center">
  <img alt="Pandas" src="https://img.shields.io/badge/-Pandas-green?style=flat&logo=Pandas&logoColor=white">
  <img alt="Tensorflow" src="https://img.shields.io/badge/-Tensorflow-orange?style=flat&logo=Tensorflow&logoColor=white">
  <img alt="Hugging Face" src="https://img.shields.io/badge/-Hugging Face-yellow?style=flat&logo=HuggingFace&logoColor=white">
</div>
<br>
<div align="center">
  
![SeGAN](https://github.com/user-attachments/assets/5a83d6f7-6225-42f2-a544-a17449c91299)


</div>

<p align="center">
SE-GAN is a framework that integrates sentiment analysis from Microsoft headlines with the Generative Adversarial Networks (GAN) model to predict the stock price of Microsoft data. The training data was retrieved from the EIKON database provided by the London Stock Exchange Group (LSEG) using API. To conduct sentiment analysis, the FINBERT model (the pre-trained model for financial text) was adopted to find a sentiment score for each headline. The final dataset includes Microsoft data from EIKON API, stock indexes from Yahoo Finance, and sentiment scores. The stock data used to train the model is between 2013-01-01 to 2023-12-31.
</p>

## Features included in the final dataset

| Columns                  | Description                                                     |
|--------------------------|-----------------------------------------------------------------|
| **Timestamp**             | Date of the trading day of the Microsoft stock                  |
| **Features**              |                                                                 |
| HIGH                      | Highest price of the stock during the day                       |
| LOW                       | Lowest price of the stock during the day                        |
| OPEN                      | Opening price of the stock for the day                          |
| COUNT                     | Number of trades executed                                       |
| VOLUME                    | Total number of shares traded                                   |
| Daily Sentiment Score     | Aggregate sentiment score from news and social media            |
| S&P 500                   | Value of the S&P 500 index                                      |
| Dow Jones                 | Value of the Dow Jones Industrial Average                        |
| NASDAQ 100                | Value of the NASDAQ 100 index                                   |
| Nikkei 225                | Value of the Nikkei 225 index                                   |
| FTSE 100                  | Value of the FTSE 100 index                                     |
| DAX 30                    | Value of the DAX 30 index                                       |
| **Label**                 |                                                                 |
| CLOSE                     | Closing price of the stock for the day                          |


## Sentiment Analysis Using FINBERT

![SeGAN_FINBERT_PIPELINE](https://github.com/user-attachments/assets/aac2418c-0540-425e-9e1e-f666e4475cbe)


To do sentiment analysis, the pipeline was created by receiving the headlines input and converting text to numbers using BERT tokenization. Then, the encoded text was fed into the FINBERT model to classify the label of the input text. The model output is a label (Positive, Neutral, or Negative) and a corresponding confidence score. 

### Example input and output data for sentiment analysis using FINBERT

| **Input Text**                                                                                       | **Output**                           |
|------------------------------------------------------------------------------------------------------|--------------------------------------|
| CallMiner Collaborates with Microsoft to Enhance AI and Machine Learning Capabilities                 | label: Positive, score: 0.9842389822006226 |
| 5 things Microsoft co-founder Bill Gates wishes he knew in his early 20s                              | label: Neutral, score: 0.9726649522781372 |
| MICROSOFT STATUS PAGE SHOWS TEAMS AND OUTLOOK.COM ARE HAVING PROBLEMS                                 | label: Negative, score: 0.9999890327453613 |

Code: [Sentiment Analysis](https://github.com/phrugsa-limbunlom/SE-GAN-FOR-STOCK-FORECASTING/blob/main/sentiment-analysis/sentiment_analysis.py) 

## Generative Adversarial Networks (GANs)

![Copy of SeGAN](https://github.com/user-attachments/assets/c445a319-383c-4a9d-857b-596e989142d6)


The Generative Adversarial Networks (GANs) model has been trained by using an iterative feedback mechanism. The generator model receives the input of stock data of the previous date and predicts the closing price of the next date. For example, if the model receives the input data from 01-01-2023 to 08-01-2023, the model will predict the closing price of 09-01-2023, and this example predicts the price from 8 sequence length. After the generator predicts the price, which is called fake price for this methodology, the predicted price (fake price) of the generator model and the real price from the training data (the closing price of 08-01-2023) will feed into the discriminator model to classify between these two prices. The generator model aims to produce a fake price that is close enough to the real price, whereas the discriminator model aims to differentiate between fake and real prices as much as possible. As such, the objective function for  optimization during training is minimizing generator loss while  maximizing discriminator loss. This technique enables the generator model to  predict the price as closely as possible to the real price from the iterative feedback process.

### Objective function of the GAN model in SE-GAN:

$$
\min_G \max_D V(G, D) = \mathbb{E}[\log D(X_{\text{real}})] + \mathbb{E}[\log(1 - D(G(X)))]
$$

$$
\min_G \max_D V(G, D) = \mathbb{E}[\log D(X_{\text{real}})] + \mathbb{E}[\log(1 - D(X_{\text{fake}})]
$$

Code: [GAN model training](https://github.com/phrugsa-limbunlom/SE-GAN-FOR-STOCK-FORECASTING/tree/main/model/GAN)

## Experimental Setup
The SE-GAN was trained using different sequence lengths including 7, 15, 30, and 60 days of input data from the previous date to predict the closing price of the next date. Also, the baseline models were trained with the same variable setting. The baseline models include LSTM and GRU due to their ability to capture the sequence input of time-series data.

## Results
The generator model of the SE-GAN was employed to predict the price from the testing data from 2024-01-01 to 2024-05-31 for evaluation. The results show that the proposed SE-GAN model can achieve lower RMSE scores than other baseline models. Also, the predicted prices yield promising results, which achieve over 69% accuracy for forecasting price change.

The graph of real (blue line) and predicted (orange line) closing prices from varying sequence lengths in the experimental setup. 


![GAN_predicted_price](https://github.com/user-attachments/assets/cc17f355-542c-4142-b250-74055ec57cf7)


The graph of real (blue line) and predicted (orange line) closing price change from varying sequence lengths in the experimental setup.


![GAN_price_change](https://github.com/user-attachments/assets/3d71f1a2-a748-4486-926d-a115a9854a7d)


## Additional Details

More details about the methodology can be read from the blog [*What you may miss about the core concept of the GAN model*](https://gifttgif.medium.com/what-you-may-miss-about-the-core-concept-of-the-gan-model-f1820d3f7efc). Also, I wrote about what I found from training a GAN model with a predictive task.

## Requirements

To install all prerequisite libraries, use the following command.
```
pip3 install -r /config/requirement.txt
```
## Acknowledgement
This repository is a part of a dissertation for the Master's degree in AI and its application in 2024 supervised by Dr. Michael Fairbank.
