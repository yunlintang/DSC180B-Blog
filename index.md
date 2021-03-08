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

In the tweets content dataset which is about covid-19, we extracted “tweet_id”, “text”, “location”, “retweeted_status”, “hashtag”, “follower_count”, “date”, and “language” from the raw dataset, and we also adjusted all columns to a suitable format and saved them as csv files. In addition, in order to have a cleaner version of the tweet text, we have converted all the text into lowercase and removed all the punctuations, stopwords, and usernames contained in the tweets. 

In the trained sentiment dataset, we only extracted “text” and “sentiment” from the original dataset, and we also made text lowercase, which can avoid standardizing the same words in different formats. Furthermore, we used “-1” to represent “negative” sentiment and “1” for “positive”, which is easier for us to calculate the total sentiment score during the 14-day period. In the daily cases dataset, we extracted cases and dates and removed all other unrelated columns. In addition, in the COVID-19 daily cases dataset, since it is relatively difficult to specify the region that the tweet users come from, we then summed up all the numbers of new cases around the world for 200 countries per day. 
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
<br/>
<br/>

![WordCloud Two](/assets/images/wc1.png){:.wcimg}
<p class="caption">Figure 1: word cloud by using CountVectorizer</p>
![WordCloud Two](/assets/images/wc2.png){:.wcimg}
<p class="caption">Figure 2: word cloud by using TfIdfVectorizer</p>


To compare the daily term frequencies and the counts of daily covid-19 cases, we tried to visualize the difference between trends by drawing the frequencies of specific terms by dates overlaid the plot of Covid-19 case numbers. After intuitively preselecting the words “great” and “sick” (two words that represent positive and negative sentiment), two graphs are generated by plotting the normalized counts of terms per day and daily case numbers from March 22 to November 30. As shown in Figures 3 and 4, we observed that there are no direct or obvious correlations between the trend of selected terms and the count of daily new cases. 
<br/><br/>

![Term Sick Frequency Plot](assets/images/sick.png){:.plotimg}
<p class="caption">Figure 3: plot of term frequencies (“sick”) overlaid by counts of new cases</p>
![Term Great Frequency Plot](assets/images/great.png){:.plotimg}
<p class="caption">Figure 4: plot of term frequencies (“great”) overlaid by counts of new cases</p>
---

### Methodology and Results

##### Baseline Model: BERT Tokenizer & Logistic regression

Before building models, we need to tokenize all text data into smaller units, so we are going to use BERT tokenizer to convert the whole text into numerical arrays.[1] By importing the transformers package, we used the “tokenizer” function to convert the text to arrays of numbers and then pad different arrays to the max length and convert it to the parse matrix. Then, we decided to use the logistic regression model as our baseline model to predict the sentiment of the
text in the covid-19 tweets dataset, and we firstly got a base accuracy of 0.53, which is well-grounded.

##### Advanced Model: CountVectorizer and SVC

Although the accuracy of the baseline logistic regression model is acceptable, it is imprecise for us to predict sentiment for the covid 19 tweets dataset. Therefore, we decided to build the SVC model as the advanced model.We are going to use the sklearn function “countvectorizer” to convert text to a vector of tokens, which can help us to use the resulting matrix as input to put it into the model. Then, when we firstly trained an SVC model with default parameter values, we got an accuracy of 0.56, which hasn't reached our goal yet. Therefore, in order to improve the accuracy, we did some parameter tuning to the SVC model. First, we just loop through different kernels which are linear, polynomial and rbf, and finally from the figure 5,we can clearly find that the SVC model using linear hyperplane achieves over 75% accuracy, but SVC models using rbf and polynomial hyperplane’s accuracy are only near to 0.5. Second, we also loop through different c values (the penalty parameter of the error term) which are 0.01, 0.1 and 1, and from Figure 6, we can find that when the c value equals to 0.1, the model reaches the highest accuracy which is 0.7782. Therefore, we decide to choose the SVC model with linear hyperplane and the c value equals 0.1 as the final advanced model which we will use to predict the sentiment for the covid-19 tweets dataset. 

<table align='center' class='imgtable'>
    <tr>
        <td style="border:none;"><img src='assets/images/svc1.png' class="wcimg"></td>
        <td style="border:none;"><img src='assets/images/svc2.png' class="wcimg"></td>
    </tr>
</table>
<p class="caption">Figure 5 (left): Accuracies of models using different kernels <br/>Figure 6 (right): Accuracies of models with different c values</p>

##### Analyzing Seasonality in Daily New Cases and Daily Tweet Sentiment

Our first step is to detrend the daily new case data. As we can see in the graph, daily cases data has upward mobility which is the result of multiple factors such as exponential transmitting rate and state policy. Our sentiment score does not have a trend in the long run. However, both of the data have a seasonality component which could be correlated. To detrend the data, we used the seasonal decompose module to locate the trends and use regression of order 3 to fit the shape of the curve. We then subtracted the composition from the original data to obtain a flat version of daily cases with only the seasonality.
<br/>
![Daily Cases Plot](assets/images/detrended_cases.png){:.plotimg2}
<p class="caption">Figure 7: Daily Cases with Detrended Daily Cases</p>

By plotting the sentiment data with the detrended data we can see that they do have similar fluctuations in the previous three months, the crest and trough of the data roughly align with one another. Along the horizontal axis, we noticed that the two time series matches less and less. One possible reason for the irregularities in later periods is that our detrended cases daily did not take into account how the upward trend affect the magnitude of the fluctuations. With the increase in cases per day, the fluctuation number also increases. On the other hand, the magnitude of the seasonality in sentiment does not vary significantly.
<br/>
![detrended Plot](assets/images/detrended.png){:.plotimg2}
<p class="caption">Figure 8: Sentiment Score VS. Detrended Daily Cases</p>

Due to the nature of time-series data, we are not exploring the causality of these two variables. Instead, we set our goals to determine the correlation of them to gain insight into how people's moods are represented by social media affected by the daily COVID cases. We first calculated the Pearson correlation of the two time series data and got a result of 0.073. This would suggest a weakly positive relationship in an independent context. In this case, both of our data are dependent on time which makes the interpretation intricate. We can only deduce that there is no strong linear relationship between Sentiment Score and Detrended Daily Cases. The cointegration test is a statistical technique that examines if two time series are integrated together at a specified degree. To perform this test, we first tested the stationarity of sentiment score and detrended cases using the Augmented Dickey-Fuller unit root test. Both of them produce a p-value of almost 0 which suggests that we should reject the null hypothesis that there is non-stationarity in the data. We accept the alternative hypothesis that our data is stationary. This effectively states that there is no cointegration between the two since both of them are stationary time series while cointegration only works on non-stationary data. 

In addition, we analyzed the seasonal frequency of the two time series. Fourier transformation decomposes a function of time into temporal frequencies. We performed the transformation and plotted the frequencies for two separate time series. Comparing the two plots, we found that Daily cases have a dominant frequency at around 0.14. Converting to days, 0.14 represents approximately 7 days which suggests daily cases mainly oscillate weekly. This is most likely related to the reporting method of daily cases. Sentiment score on the other hand does not have a dominant frequency. It has multiple spikes with similar magnitudes.

<table align='center' class='imgtable2'>
    <tr>
        <td style="border:none;"><img src='assets/images/sentiment.png' class="wcimg"></td>
        <td style="border:none;"><img src='assets/images/cases.png' class="wcimg"></td>
    </tr>
</table>
<p class="caption">Figure 9 (left): Sentiment Score Decomposition<br/>Figure 10 (right): Daily Cases Decomposition</p>

---

### Findings and Conclusions

Covid-19, as one of the biggest problems facing humans in this century, has affected all aspects of people’s lives, from the way how people interact, to people’s lifestyles, and even social media. In this project, by using the “covid-19 tweets” dataset obtained from the Panacea Lab, we predicted the total sentiment score of each day’s tweets from March 22 to November 30 (inclusive) and explored the correlation between sentiment scores and daily new cases of Covid-19. In order to predict the sentiment score of the Covid-19 tweets dataset, we found a pre-trained tweets dataset with labeled sentiment scores on the Kaggle website. After model evaluation and parameter tuning, we finally decided to use the SVC model(c = 0.1, kernel = ‘linear’) which has an accuracy of 0.7782 to predict the sentiment score of the Covid-19 tweets dataset. 

To compare the time series of daily sentiment scores and detrended daily cases, we plotted them on the same graph. Although we observe some initial correspondence between the two variables, in order to make sure they are statistically significant, we conducted multiple tests including Pearson Correlation, Augmented Dickey-Fuller unit root test, and Fourier transform. We used Pearson Correlation to get an approximate idea of the overall correspondence between sentiment and daily cases since the test is subject to the effect of noise in time series. A score of 0.073 suggests that there might be a weak positive relationship. After that, we looked into cointegration tests which check whether two time series are integrated in a way that does not change in the long run. The result suggests that there exists no cointegration relationship since both of our time series are statistically significant stationary. 

Since the previous two tests ruled out relationships between our variables. We studied features that differentiate them by using Fourier Transformation which decomposes temporal frequencies out of time series. By plotting the frequencies, we noticed that sentiment score has no dominant frequency while daily cases have one at 0.14 which corresponds to a period of roughly 7 days. This could explain why the first two tests do not find a strong relationship between them. If the frequency of the two does not coincide with each other, these two time series are constantly out of phase which results in low statistical significance in correlation. The weekly period of the daily cases is likely connected to the reporting mechanism. For example, some facilities may choose to report their weekly cases on Monday instead of reporting daily. While sentiment score is calculated on a continuous daily basis, the discrepancy resulted in different oscillating frequencies of the seasonality in daily cases and sentiment.


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