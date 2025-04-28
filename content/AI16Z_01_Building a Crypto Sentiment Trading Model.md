---
Title: Building a Crypto Sentiment Trading Model (by Group "AI16Z")
Date: 2025-03-23 16:00
Category: Reflective Report
Tags: Group AI16Z
---

By Group "AI16Z"

## How We Chose Our Topic

Sentiment analysis is quite a strong financial instrument because it allows traders and investors to evaluate market moods via the analysis of news, social media, and other information. Of course, our passion for trading and investing led us to chase sentiment analysis as a way to dive into how markets behave. Noticing its potential, we then settled on where in the market it would be most effective.

Unlike conventional financial markets that are driven by fundamental indicators such as economic news and company earnings, the cryptocurrency market is very much influenced by sentiment. Market movement is often influenced by such events as news reports, social media posts, and tweets from influential figures as well as discussion forums online. Because of the decentralized and speculative nature of crypto assets, price action is dictated by market sentiment.

Such an extreme reliance on sentiment makes the cryptocurrency market a flawless setup to analyze sentiment. By using complex natural language processing (NLP) techniques, machine learning software, and statistical analysis, we can measure the sentiment in markets systematically and objectively in real time. It helps us pick out trends, foresee market trends, and trade smartly. Noticing these advantages, we have opted to apply sentiment analysis in the crypto market specifically, where it can be of valuable assistance in identifying investor sentiment and predicting price action.

## Data Collection

The first task is identifying reliable sources containing text data related to Bitcoin. The sources should also include the hidden sentiment in the market. At first, we discussed using both news articles and social media posts, such as Twitter, to capture a more general view of market sentiment. However, after further research, we realized that there are several obstacles to using social media data, such as the difficulty of filtering bot-generated posts, and noises. For example:

- **Bot noise** – 30% of tweets from trending crypto hashtags were spam.
- **Context ambiguity** – Phrases like “This coin is fire!” could mean success or disaster.

Therefore, we decided to focus only on news articles, which are more structured and with credible sources, for sentiment analysis.

After evaluating several news sources, we eliminated a bunch of news platforms that provide limited Bitcoin-related articles. We finally selected Coindesk and NewsAPI owing to their API accessibility and extensive coverage of cryptocurrency-related news. These data sources provide historical Bitcoin-related news articles, allowing us to do a time-series analysis which is important for predicting future prices of Bitcoin.

Here's an example of how we retrieve Coindesk news articles via API:

```python
url = 'https://data-api.coindesk.com/news/v1/article/list'
headers = {"Content-type":"application/json; charset=UTF-8"}
api_key = '<API_KEY>'
to_ts = int(datetime.datetime.now().timestamp())
param = {
    'api_key': api_key,
    'lang': 'EN',
    'categories': 'BTC',
    'to_ts': to_ts
}
response = requests.get(url, headers=headers, params=param)
json_data = response.json()
```

## Potential Models & Training

During the initial phase of our project, we debated between using a lexicon-based approach (such as VADER) and machine learning models for data analysis. Both models exhibit pros and cons. The lexicon-based method was quick and interpretable; however, it struggled with detecting sarcasm and context-specific sentiment in financial news. On the other hand, training machine learning models provided greater accuracy but required labelled datasets, which were difficult to obtain. To balance these trade-offs, we decided on a hybrid approach—using lexicons to establish a baseline sentiment score and refining these results with machine learning models trained on labelled financial news.

Our current focus involves selecting the optimal machine learning model from a suite of candidates, including Logistic Regression, KNN, Naïve Bayes, QDA, and LDA. We plan to develop the Python code for these models first, test them over a minimum of 10 trading days, and select the one with the highest average accuracy. We aim to finalize the Python code for these models before April and begin testing the results with time-series market data. This integration will allow us to analyze how sentiment trends correlate with price movements or trading volumes. Our **Blog 2** would document daily model performances for transparency, while the final presentation will prioritize a summary of overall accuracy and practical insights due to time constraints.

## Potential Limitation of Our Method

Our hybrid approach encounters the project’s compressed timeline. While testing models over 10 trading days provides a snapshot of performance, this short window may not adequately account for broader economic cycles—such as bull or bear markets—that inherently influence sentiment patterns. As a result, sudden volatility, such as geopolitical events could affect our model. We aim to mitigate this by explicitly acknowledging the economic context of our testing period in **Blog 2** and cautioning against overgeneralizing our findings to all market conditions.
