---
Title: Blog 1 (by Group "NLPredict")
Date: 2025-03-21 18:00
Category: Reflective Report
Tags: Group NLPredict
---

Introduction

The following blog covers the developmental progress of NLPredict by March 21st 2025 and will explore the various checkpoints, difficulties and breakthroughs our team has made throughout the developmental process.

Currently our team has finished planning and programming the early phase of NLPredict’s model. We have also planned for certain goals we aim to reach by the end of the current phase of development, namely to have finished research into the current financial market as well as beginning analysis and testing different model architectures.

Market Research

Research into the market was conducted by Jason, we note that there have been previous cases of using NLP and Sentiment Analysis as supporting but not mainstream predictors of stock prices. Presently, we understand that NLPredict is not a unique product in the market, in truth, there are many competitors of NLPredict in this regard. Due to our lack of distinction from other products in the market, we are considering elements which could allow us to stand out. In order to create distinction, we considered doing so from either an analytical perspective, focusing on making more precise predictions, or from a creative approach, focusing more on the elements of NLPredict’s creation process. One of the proposed key elements to making our product different creatively is the reliance and emphasis on using NLP and Sentiment Analysis instead of other data types such as numerical data from stock prices.

Currently, for simplicity, we are using Yahoo Finance and Finviz and are scraping the articles’ text data as shown below:

def get_news(ticker, company_name, days=30):
    """Collect financial news articles for the specified company with improved error handling."""
    print(f"Collecting news for {company_name} ({ticker})")
    news_items = []
    
    # Track success of each source
    source_success = {
        'yahoo': False,
        'marketwatch': False,
        'seekingalpha': False,
        'finviz': False
    }
    
    # 1. Try Yahoo Finance with improved headers and selectors
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }

The above code was used in order to obtain data from Yahoo Finance. Currently we have multiple such implementations of the above 'get_news' functions on various news sources. However, we want to find a more efficient method of increasing the number of inputs without having to increase the amount of code drastically. 

Current Development

The development of NLPredict has been going smoothly. Herbert has compiled the source code required for data extraction, data modelling, data visualisation and feature engineering. The source code has undergone a few tests but the results have been less than ideal, thus there is still a need for editing the code. Presently, the code is still very primitive and is not completely optimised for the consumer to understand or use especially on devices with less memory or computing power. This is within expectations as the initial plan amongst our team was to create the product first before making it explainable and widely accessible to the market.

An example of such code can be found below. The following code shows the change in market sentiment, the level of which was obtained from our Sentiment Analysis, over time. This allows for a direct visual comparison with the Stock Market at the time. In the source code, we have made multiple such visualisations each with different functions.

def visualize_results(data, model):
    """Visualize sentiment vs. returns and predictions."""
    print("Generating visualizations...")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Sentiment Time Series
    plt.subplot(2, 2, 1)
    plt.plot(data.index, data['Sentiment'], 'b-', label='Sentiment')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Score Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

During the testing of the model, it was also noted that some level of additional data cleaning was required as well as adjustment of the hyperparameters of the model. This will be our upcoming focus as we aim to be able to gain more accurate and reliable results from the model first before tuning it for the long term. We have also encountered problems with acquiring data from certain stocks which may also be a matter for data preprocessing. Additionally, the model still has a time lag when compared to the rate at which news enters the market, as such, some level of preliminary prediction may be necessary to account for this duration.

Current Objectives

Our team is now entering discussions about increasing the variety of data sources as well as looking for other data sources to supplement our Sentiment Analysis. We are also looking at different metrics to gauge the accuracy of our model’s predictions. Other considerations such as using numerical data in contandem with textual data have also been proposed. Currently, our stance on this matter is to avoid using numerical data if possible, but to include it as support if it aids with the model’s predictions.

Presently, we have not made significant breakthroughs outside of progressing our main tasks as a group. We have had some difficulties in reaching our goals within the given time period, mainly due to the scope of our project. While we have discussed the approach and whether or not we would like to revise our scope and approach, we have ultimately decided against doing so in favour of creating a more robust product.

To conclude, we would like to offer some insight on the projections our group has for the future of this project. Looking forward, aside from the various issues touched on previously, we are aiming to focusing on finding any anomalies as well as potential points of error which may affect the accuracy of our predictions from a consumer's perspective. In particular, we would like to find some stocks which may not be as predictable with Sentiment Analysis than other stocks. Moreover, we would like to create a more user-friendly aspect to our model, making it more accessible not only from a technical perspective but also from a convenience perspective.