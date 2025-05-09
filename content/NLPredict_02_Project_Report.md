---
Title: Blog 2 (by Group "NLPredict")
Date: 2025-04-20 20:00
Category: Reflective Report
Tags: Group NLPredict
---

# Introduction

The following blog covers the developmental progress of NLPredict by April 21st 2025 and will explore the various checkpoints, difficulties and breakthroughs our team has made throughout the developmental process. In addition, this blog will also reflect on various mistakes made within NLPredict’s programming as well as various challenges that our team has faced outside of the technical aspects of this project.

# Progress since the Previous Blog

Currently our team has finished programming and implementing the final phases of NLPredict’s model, achieving a much higher accuracy in prediction than our initial values. We have also achieved most of our targets in terms of backtesting and evaluating NLPredict’s model, both from an accuracy and efficiency perspective. Our research into the market has been helpful in narrowing down the sources of errors as well as determining whether or not certain stocks and data sources should be included into training the model.

NLPredict’s model has now incorporated the use of other pieces of financial data such as the Standard and Poor 500’s (S&P 500) historical price, as well as various indices such as the Hang Seng Index (HSI) in order to corroborate and verify the accuracy of NLPredict’s predictions. Initially, there was some discussion as to whether or not this decision would go against the goal of making Natural Language Processing (NLP) and Sentiment Analysis play a more active role in making stock price predictions. However, after a lengthy discussion, it was decided that using other pieces of financial data in support of NLP did not go against the original aim of this project.

Jason has been researching the market and modern uses of Sentiment Analysis which has helped with determining the effect of market sentiment on the prices of Stocks, as well as recording certain tasks which needed to be done for the project. This was particularly useful when considering short-term price-shocks caused by sudden events or worries (such as natural disasters) to decrease or increase the effects of market sentiment on particular stocks. Using this information as an opportunity, we have also discussed how to deal with such anomalies in the market and our data sources. Initially our team wanted to leave the problem as is, given it was not a problem affecting a large number of stocks. However, after some reconsideration, it was decided to make some minor changes to the model’s data inputs in favor of a more complete final product.

Herbert has been mostly in charge of programming and fine tuning our model while the rest of our team has been working on backtesting and debugging the model. During this process, we have looked at using different accuracy metrics to determine the validity of NLPredict’s predictions. Ultimately, we have determined that using mainly financial and regression metrics would be the most efficient as it would be the easiest to explain and therefore the most accessible to the wider market.

A sample of the code we have used can be found below.

```python
def evaluate_predictions(y_true, y_pred, model_type='regression'):
    """Evaluate predictions with comprehensive metrics"""
    metrics = {}
    
    if model_type == 'regression':
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Financial specific metrics
        metrics['accuracy_direction'] = np.mean((y_true * y_pred) > 0)  # Direction accuracy
        metrics['mean_return'] = np.mean(y_pred)
        metrics['sign_consistency'] = np.mean(np.sign(y_pred) == np.sign(y_true))
        
        # Risk-adjusted metrics
        metrics['sharpe'] = np.mean(y_pred) / np.std(y_pred) if np.std(y_pred) > 0 else 0
            
    else:
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1'] = f1_score(y_true, y_pred, average='binary')
        
        # Class balance
        metrics['positive_ratio'] = np.mean(y_true)
        metrics['predicted_positive_ratio'] = np.mean(y_pred)
    
    return metrics
```

While initially it seemed that using only two indicators of accuracy was insufficient and that having more indicators would give a more holistic view of the model’s accuracy, in hindsight, it seems that having fewer indicators has aided us in judging whether the model’s predictions were sufficient.

A key development of the project was the narrowing down of data sources that we have incorporated into the NLPredict’s data input. Initially, NLPredict was taking in data from various sources. However, this came with the problem that there were certain repetitions that were entering its predictions which led it to favour one sentiment in particular. As such, the decision was made to narrow down its data input to be from mainstream sources such as Yahoo Finance, rather than various financial news sources. While this decision was made, it was agreed that it was a temporary solution rather than a permanent one. The team has unanimously recognised that the benefits in an increase in data sources and inputs would far outweigh the demerits. However, it was noted that it would require much more engineering and testing to determine whether or not a data source or input was valuable or useful for NLPredict’s Training and predictions.

The initial code which uses various sources can be found below with most of the code omitted in order to demonstrate the breadth of sources used. The following list is an example of some of the sources used.

```python
def get_news_from_finviz(ticker, max_articles=40):
    """Get news from Finviz with robust parsing"""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    logger.info(f"Fetching news from Finviz: {url}")

def get_news_from_wsj(ticker, max_articles=20):
    """Get news from Wall Street Journal Search"""
    url = f"https://www.wsj.com/search?query={ticker}"
    logger.info(f"Fetching news from WSJ: {url}")

def get_news_from_marketwatch(ticker, max_articles=20):
    """Get news from MarketWatch"""
    url = f"https://www.marketwatch.com/investing/stock/{ticker}"
    logger.info(f"Fetching news from MarketWatch: {url}")

def get_news_from_reuters(company_name, max_articles=20):
    """Get news from Reuters"""
    url = f"https://www.reuters.com/search/news?blob={quote_plus(company_name)}"
    logger.info(f"Fetching news from Reuters: {url}")

def get_news_from_bloomberg(company_name, max_articles=15):
    """Get news from Bloomberg (note: might have limited success due to paywall)"""
    url = f"https://www.bloomberg.com/search?query={quote_plus(company_name)}"
    logger.info(f"Fetching news from Bloomberg: {url}")

def get_news_from_financial_times(ticker, company_name, max_articles=15):
    """Get news from Financial Times"""
    # Try company name first, then ticker
    url = f"https://www.ft.com/search?q={quote_plus(company_name)}"
    logger.info(f"Fetching news from Financial Times: {url}")
```

After discussion and testing, we have narrowed down the data sources, some of which can be found below.

```python
def get_news_from_yahoo_finance(ticker, max_articles=40):
    """Get news from Yahoo Finance with robust parsing"""
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    logger.info(f"Fetching news from Yahoo Finance: {url}")

def get_news_from_bloomberg(company_name, max_articles=15):
    """Get news from Bloomberg (note: might have limited success due to paywall)"""
    url = f"https://www.bloomberg.com/search?query={quote_plus(company_name)}"
    logger.info(f"Fetching news from Bloomberg: {url}")

def get_news_from_financial_times(ticker, company_name, max_articles=15):
    """Get news from Financial Times"""
    # Try company name first, then ticker
    url = f"https://www.ft.com/search?q={quote_plus(company_name)}"
    logger.info(f"Fetching news from Financial Times: {url}")
```

Looking forward, some other developments we would like to make would be to incorporate more data types that would accentuate the use of Sentiment Analysis and NLP in stock price predictions. Currently, development on this front has been low due to a lack of ideas regarding such data outside of numerical and historical data. Despite this, the team has been satisfied with the work that was accomplished on this project in the time given.