# SE-GAN-FOR-STOCK-FORECASTING
This repository is part of a dissertation for a Master's degree in AI and its application in 2024

## SE-GAN: Sentiment-Enhanced GAN for Stock Price Forecasting - A Comprehensive Analysis of Short-Term Prediction

![image](https://github.com/user-attachments/assets/f580fe0a-8418-412c-9064-cba8db7b4f36)

Conducted sentiment analysis on Microsoft headlines using the FINBERT model (pre-trained NLP model) from Hugging Face and integrated the sentiment scores with the Microsoft stock data from the EIKON database by LESEG along with the historical stock index from Yahoo Finance. The final dataset was trained by the Generative Adversarial Networks (GANs) for predicting the daily closing prices of Microsoft Stock.

### Sentiment Analysis Using FINBERT

![image](https://github.com/user-attachments/assets/940bfc9f-bac2-4a97-b33e-5036e57493b7) 

The pre-trained model, FINBERT, was employed to convert the headlines to sentiment scores. The model received headlines as the input data before feeding them to the pipeline to convert the text data, by using BERT tokenization, into encoded numerical representations that the model can interpret. The encoded data is then fed into the FINBERT model, a BERT-based classifier fine-tuned for financial text, to classify the label of input text. Finally, the model outputs a label (Positive, Neutral, or Negative) and a corresponding confidence score. 

### Generative Adversarial Networks (GANs)

The Generative Adversarial Networks (GANs) model has been trained through an iterative feedback mechanism. The process commences with employing a generator model to generate closing prices, which are subsequently input into the discriminator model to receive feedback regarding the authenticity of the prices generated. Once the generator model receives feedback from the discriminator, the model will undergo optimization to generate more synthetic prices to fool the discriminator, aiming to generate prices that closely approximate actual market values.

![image](https://github.com/user-attachments/assets/ba4b8ddc-2bb5-44e2-bce2-a93915606237)

The results show that the proposed SE-GAN model can achieve lower RMSE scores than other baseline models like LSTM and GRU. Also, the predicted prices yield promising results, achieving over 69% accuracy for forecasting price change.

## Additional Details

### What you may miss about the core concept of the GAN model
More details about the methodology can be read from the blog below. Also, I wrote about what I found from training GAN with a predictive task.
https://gifttgif.medium.com/what-you-may-miss-about-the-core-concept-of-the-gan-model-f1820d3f7efc

or full detail in the report: [CE901_Dissertation_2311569.pdf](https://github.com/phrugsa-limbunlom/SE-GAN-FOR-STOCK-FORECASTING/blob/main/CE901_Dissertation_2311569.pdf) or presentation: [Dissertation_PDO_2311569.pptx](https://github.com/phrugsa-limbunlom/SE-GAN-FOR-STOCK-FORECASTING/blob/main/Dissertation_PDO_2311569.pptx)
