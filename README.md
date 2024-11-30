# SE-GAN-FOR-STOCK-FORECASTING
This repository is part of a dissertation for a Master's degree in AI and its application in 2024

## SE-GAN: Sentiment-Enhanced GAN for Stock Price Forecasting - A Comprehensive Analysis of Short-Term Prediction

![image](https://github.com/user-attachments/assets/f580fe0a-8418-412c-9064-cba8db7b4f36)

Conducted sentiment analysis on Microsoft headlines using the FINBERT model (pre-trained NLP model) from Hugging Face and integrated the sentiment scores with the Microsoft stock data to train the Generative Adversarial Networks model (GANs) for predicting daily closing prices of Microsoft Stock.

### Sentiment Analysis Using FINBERT

![image](https://github.com/user-attachments/assets/940bfc9f-bac2-4a97-b33e-5036e57493b7) 

The pre-trained model, FINBERT, was employed to convert the headlines to sentiment scores. The model received headlines as the input data before feeding them to the pipeline to convert the text data, by using BERT tokenization, into encoded numerical representations that the model can interpret. The encoded data is then fed into the FINBERT model, a BERT-based classifier fine-tuned for financial text, to classify the label of input text. Finally, the model outputs a label (Positive, Neutral, or Negative) and a corresponding confidence score. 

### Generative Adversarial Networks (GANs)

The Generative Adversarial Networks (GANs) model has been trained through an iterative feedback mechanism. The process commences with employing a generator model to generate closing prices, which are subsequently input into the discriminator model to receive feedback regarding the authenticity of the prices generated. Once the generator model receives feedback from the discriminator, the model will undergo optimization to generate more synthetic prices to fool the discriminator, aiming to generate prices that closely
approximate actual market values.

![image](https://github.com/user-attachments/assets/ba4b8ddc-2bb5-44e2-bce2-a93915606237)

The results show that the proposed SE-GAN model can achieve lower RMSE scores than other baseline models like LSTM and GRU. Also, the predicted prices yield promising results, achieving over 69% accuracy for forecasting price change.

## What you may miss about the core concept of the GAN model
https://gifttgif.medium.com/what-you-may-miss-about-the-core-concept-of-the-gan-model-f1820d3f7efc
