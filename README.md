# 📰NEWS CLASSIFICATION🗞️

## 📝Problem Statement
In today’s world, data is power. With News companies having terabytes of data stored in
servers, everyone is in the quest to discover insights that add value to the organization.
With various examples to quote in which analytics is being used to drive actions, one that
stands out is news article classification.
Nowadays on the Internet there are a lot of sources that generate immense amounts of
daily news. In addition, the demand for information by users has been growing
continuously, so it is crucial that the news is classified to allow users to access the
information of interest quickly and effectively. This way, the machine learning model for
automated news classification could be used to identify topics of untracked news and/or
make individual suggestions based on the user’s prior interests.

## 📅Dataset
The dataset is download from [Kaggle](https://www.kaggle.com/c/learn-ai-bbc/data)
### Data Description
* **ArticleID**: Unique Id for each news
* **Text**: Particular text of the header and article.
* **Category**: Category of the article(Tech, Business, Sport, Entertainment, politics)
![33](https://user-images.githubusercontent.com/66157611/204429311-e4d9f0b6-2c85-45cf-a883-6805d8ff9e04.png)

## Data Preprocessing
Our data is not preprocessed and it contains lot of punctuations and numbes. To convert of raw data as preprocessed format(like removing punctuations, numbers, single character, multiple spaces). We had created a function `preprocess_text`.

After the raw text is preprocessed, we have to encode our labels to numerical encoding as there are categorical encoded before. For this we are using **OneHotEncoder** and **LabelEncoder**.

## Modelling
For modelling we have used six different models(including Baseline).
* Baseline
* Simple dense model
* Conv1D
* LSTM
* GRU
* Bidirectional 
Out of all these six models **Conv1D** outperformed the rest.
![all_models](https://user-images.githubusercontent.com/66157611/204430568-5b52ec2e-19c4-4f0c-9cf1-9d7a38ad9c71.png)


## Evaluation
For evaluation we have used classification metrics(Accuracy score, Precision Score, Recall Score, F1-score)
