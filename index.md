---
layout: default
title: "COVID-19 Sentiment Analysis on Social Media"
---

## COVID-19 Sentiment Analysis on Social Media

Data Science Capstone Project - DSC180AB B02

Developed by Yunlin Tang, Jiawei Zheng, Zhou Li

---

### Introduction

Covid-19 changed everyone, from the way we interact, to how we work, and our methods of communication, especially through social media. During this pandemic period, social media becomes a huge and important part of people’s daily lives. It provides mobile users a convenient way to connect to each other around the world and acquire updated and trending information about the topic of covid-19. Besides, people can also express their thoughts and feelings toward certain topics by posting on social media. Throughout the studying of this quarter, we noticed that there are numbers of posts in our Twitter dataset that are related to the topic of covid-19 having some strong emotions and sentiments. In the meantime, a previous study[1] has shown that more people are experiencing negative emotions such as anxiety and panic under this pandemic period. Therefore, we are interested in analyzing the posts that are related to the topic of covid-19 on social media and investigating the emotions of the results implied in these posts will lead to. 

We start our investigation using the “covid-19 tweets” dataset obtained from the Panacea Lab[3] by performing sentiment analysis on the tweet text. Sentiment analysis and opinion mining are useful in the sense that it contributes to the understanding of human emotions by observing people’s engagement in social platforms. Using social media, we are able to monitor the user’s feed with sentiment analysis. For the purpose of this project, we expect that the results can answer the potential investigating question:  “How is the trend of daily sentiment related to the change in the number of daily COVID cases?”. The motivation behind this question is that Tweet sentiments can be analyzed in real-time with relatively minor effort, but COVID case data requires huge amounts of human and economic resources to obtain. We will build a predictive model for daily new cases of covid-19 using sentiments from the previous time period. Having a reliable and efficient model that predicts the daily cases can help with the containment of the pandemic.

---

### Datasets

##### Data Collection

There are three datasets obtained for this project. First, we used the dataset which includes the daily Tweets IDs which can regenerate tweets about the covid-19 from March 22 to November 30 (inclusive) in the year 2020, collected by the [Panacea Lab at Georgia State](https://github.com/thepanacealab/covid19_twitter). Then we sampled at a rate of one out of 360 tweet IDs per day for convenience purposes. On this subsampled dataset, we performed the Twitter collection process by using the Twitter API function “twarc” to rehydrate which requests the full tweet content based on the tweet IDs, and then got all the tweets about the covid-19. 

After obtaining all the tweets about covid-19 in the subsampled dataset, a training dataset that includes similar tweet contents with sentiment labels will be required in order to build the prediction models for sentiment analysis in the covid-19 tweets dataset. We then found a dataset from [Kaggle](https://www.kaggle.com/kazanova/sentiment140) that contains 1.6 million training data, and each row in the dataset contains the text of a tweet and a sentiment label, which the text variable can be extracted as the feature and sentiment label as the output result to make predictions to sentiment.

In order to observe and determine the significant correlation between the sentiment in the  covid-19 related social media posts and the numbers of disease daily cases, we obtained a  dataset which contains daily new positive cases and death cases all over the world, which is from [Our World in Data](https://github.com/owid/covid-19-data/tree/master/public/data)[4]. 


##### Data Processing and Cleaning

In the tweets content dataset which is about covid-19, we extracted “tweet_id”, “text”, “location”, “retweeted status”, “hashtag”, “follower_count”, “date”, “language” and “text” from the raw dataset, and we also adjusted all columns to a suitable format and saved them as csv files. In addition, in order to have a cleaner version of the tweet text, we have converted all the text into lowercase and removed all the punctuations, stopwords, and usernames contained in the tweets. 

In the trained sentiment dataset, we extracted “id”, “text” and “sentiment” from the original dataset, and we also made text lowercase, which can help to standardize the words in different formats. Furthermore, we used “-1” to represent “negative” sentiment and “1” for “positive”, which is easier for us to calculate the total sentiment score during the 14-day period. In the daily cases dataset, we extracted cases and dates and removed all other unrelated columns. In addition, 
in the COVID-19 daily cases dataset, since it is relatively difficult to specify the region that the tweet users come from, we then summed up all the numbers of new cases around the world for 200 countries per day. 
<br/>

<div class="tables" markdown="1">

|       | Number of Observations | Avg. Text Length  | Median Text Length | Avg. Follower Counts |
|-------|------------------------|-------------------|--------------------|----------------------|
| Stats | 1,007,496              | 87.639            | 85.0               | 19,146.348           |

*Table 1: Statistics of Tweet Contents Dataset*
<br/>

|       | Number of Observations | Avg. Text Length  | Median Text Length | Counts of Pos. Sentiment |
|-------|------------------------|-------------------|--------------------|--------------------------|
| Stats | 1,600,000              | 74.090            | 69.0               | 800,000                  |

*Table 2: Statistics of Trained Sentiment Dataset*
<br/>

|       | Number of Observations | Avg. Daily New Cases | Std. of Daily New Cases | Median of Daily New Cases |
|-------|------------------------|----------------------|-------------------------|---------------------------|
| Stats | 254                    | 496,645.024          | 322,896.980             | 457,871                   |

*Table 3: Statistics of Daily New Cases Dataset*
<br/>

</div>

---

### Data Analysis

##### Text Analysis

After cleaning the text in the Twitter dataset, we have performed an exploratory data analysis on it. By calculating the term frequency and Tf-Idf throughout these Twitter posts, the tables of frequencies were acquired respectively. We noticed that these two vectorizers gave out similar results; for example, the three most frequent terms in both tables are “covid19”, “coronavirus”, and “trump”. In order to visually compare the results, a graph of the word cloud for both vectorizers was generated as shown in Figures 1 and 2 below.
![WordCloud Two](/assets/images/wc1.png){:.wcimg}
<p class="caption">Figure 1: word cloud by using CountVectorizer</p>
![WordCloud Two](/assets/images/wc2.png){:.wcimg}
<p class="caption">Figure 2: word cloud by using TfIdfVectorizer</p>


To compare the daily term frequencies and the counts of daily covid-19 cases, we tried to visualize the difference between trends by drawing the frequencies of specific terms by dates overlaid the plot of Covid-19 case numbers. After intuitively preselecting the words “great” and “sick” (two words that represent positive and negative sentiment), two graphs are generated by plotting the normalized counts of terms per day and daily case numbers from March 22 to November 30. As shown in Figures 3 and 4, we observed that there are no direct or obvious correlations between the trend of selected terms and the count of daily new cases. 

---

### Methodology

##### CountVectorizer and SVC

First, we used the sklearn function “countvectorizer” to convert text to a vector of tokens, which can help us to use the resulting matrix as input to put it into the model. Then, we trained an SVC model to predict the sentiment of tweet contents, and select the  “c” value as 0.01.  Finally, we got an accuracy of 0.53. 

##### BERT Tokenizer & Logistic Regression

Besides using the SVC model, we are also going to use BERT tokenizer and logistic regression to predict the sentiment of the text in the covid-19 tweets dataset[1]. By importing the transformers package, we used the “tokenizer” function to convert the text to arrays of numbers and then pad different arrays to the max length and convert it to the parse matrix. Then, we built the logistic regression model to predict the sentiment of the text. Finally, we got an accuracy of 0.56. 

##### NLTK Vader

To explore the sentiment with a more comprehensive perspective, we fitted Vader in the nltk module to the dataset. It does not require us to tokenize the sentences since it automatically recognized words within its dictionary. The efficiency allows us to obtain a compound score that shows the comprehensive sentiment of the sentence tweet in this scenario.

With the score plotted, we can see that there exists fluctuation in the sentiment between days but most of the time the score stays below 0 which suggests that the overall mood of the COVID-19 tweets is negative. 

##### Modeling Seasonality in Daily New Cases and Daily Tweet Sentiment (from Vader)

Our first step is to detrend the daily new case data. As we can see in the graph, daily cases are climbing every data which are the result of multiple factors such as exponential transmitting rate and state policy. Our sentiment score does not have a trend in the long run. However, both of the data have a seasonality component which could be correlated. To detrend the data, we used the seasonal decompose module to locate the trends and use a regression of order 3 to compose the shape of the curve. We then subtracted the composition from the original data to obtain a flat version of daily cases with only the seasonality. By plotting the sentiment data with the detrended data we can see that they do have similar fluctuations before 2020/10/18, the crest and trough of the data roughly align with one another. One possible reason for the irregularities in later periods is that our detrended cases daily did not take into account how the trend affected the magnitude of the fluctuations. With the increase in cases per day, the fluctuation number also increases.

Our main goal is a multivariable time series prediction task. We are using past daily cases and tweet sentiments to predict future cases. Due to the nature of time-series data, we are not exploring the causality of these two variables. Instead, we are keen to determine the correlation of them so that we can build a predictive model that predicts daily cases from daily sentiment. To include both time series data and the extra variable, we decided to transform our time-series daily case data to fit supervised machine learning which will open up tons of options for models. By shifting the data, we created multiple columns that represent the neighboring time periods of the corresponding entry and used the last column as our prediction variable. In this case, we got rid of the first 15 rows of the data to make sure each row has the previous 14 units as predictors. We then added the daily sentiment score as the extra predictor to make sure we are predicting on both the time series and the tweet sentiment. The train and test data are also 15 time-unit apart to make sure there is no overlapping data that would lead to over-optimistic scores.

The next step we took is to find the best model that can predict the daily case fluctuations. We tried some of the most common machine learning algorithms such as linear regression, SVC, and decision tree. Many of them did not perform very well likely due to the oscillation nature of our data. Besides, the scarcity of the data stops the models from capturing the full trend of the data. Among which, KnnRegression achieved a root mean square of around 21000.

A remedy for the problem above is the LSTM model. The Long Short Term Neural Network is specifically designed to memorize the long term behavior which is suitable for the regular oscillation in our data. It also allows for multiple variable inputs which are not common in time series models. We used only one hidden layer in the model because the correlation between our variables is not complicated logically. We used a function of date as the internal state of the model as LSTM resets the internal state on every epoch. Since our data size is relatively small, we started with a large epoch and small batch size to begin our grid search. By plotting the training-validation loss, we can monitor and find the best parameters for our model.

---

### Reference

- Koyel Chakraborty, Surbhi Bhatia, Siddhartha Bhattacharyya, Jan Platos, Rajib Bag, Aboul Ella Hassanien, Sentiment Analysis of COVID-19 tweets by Deep Learning Classifiers—A study to show how popularity is affecting accuracy in social media, Applied Soft Computing, Volume 97, Part A, 2020, 106754, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2020.106754.

- Michela Del Vicarioa, Alessandro Bessib, Fabiana Zolloa, Fabio Petronic, Antonio Scalaa,d, Guido Caldarellia,d, H. Eugene Stanleye, and Walter Quattrociocchia,1 (2015) The spreading of misinformation online.

- Jmbanda, covid19_twitter(2020), GitHub repository, https://github.com/thepanacealab/cov
id19_twitter/tree/master/dailies

- edomt, Data on COVID-19 (coronavirus) by Our World in Data, GitHub repository, https://github.com/owid/covid-19-data/tree/master/public/data

- Jason Brownlee, Multivariate Time Series Forecasting with LSTMs in Keras, Machine Learning Mastery, https://machinelearningmastery.com/multivariate-time-series-forecasting-lst
ms-keras/

---