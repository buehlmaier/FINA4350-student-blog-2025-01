---
Title: Overcoming Sentiment Analysis Challenges with FinBERT and SpaCy (by Group "FinText")
Date: 2025-03-14 00:00
Category: Reflective Report
Tags: Group FinText
---

## Introduction

Information is the foundation of decision-making in the fast-paced and ever-changing financial markets. From news articles and social media updates to earnings reports and analyst opinions, the constant stream of information provides investors with critical insights. However, the sheer volume and complexity of this information present significant challenges. How can we extract actionable insights from this deluge of data? This question guided our group’s exploration of Natural Language Processing (NLP) techniques for financial text analysis.

Our project, "News-driven NLP Quant," aims to analyze financial news to provide insights that inform trading strategies and optimize portfolio performance. As we worked on the project, we encountered several challenges that shaped our approach and led us to adopt domain-specific
models like FinBERT and feature extraction techniques such as Named-Entity Recognition (NER) using SpaCy.

## The Challenge of TextBlob Sentiment Analysis in Financial Texts

Sentiment analysis emerged as a natural starting point for our project. By assessing the tone of financial news, we could quantify market sentiment and help investors make informed decisions. However, as we delved deeper, we realized that sentiment analysis in the financial domain is far from straightforward.

Initially, we experimented with TextBlob, a popular Python library for sentiment analysis. While TextBlob is easy to use and provides a quick way to determine whether text has a positive, negative, or neutral tone, it has significant limitations when applied to financial texts. TextBlob
treats each word independently, lacking the ability to understand the contextual meaning of words. This is particularly problematic in financial texts, where domain-specific jargon and ambiguous phrases are common. Additionally, TextBlob’s general-purpose approach results in lower accuracy when analyzing the nuanced language of financial markets.

Here’s an example of how we used TextBlob for sentiment analysis, which returns a polarity score based on the sentiment of the news article.

``` Python
from textblob import TextBlob
print(TextBlob(news_article).polarity)
```

While TextBlob provided a quick and easy way to gauge sentiment, its limitations led us to explore more advanced solutions.

## Enhancing Sentiment Analysis with FinBERT

To address the shortcomings of traditional sentiment analysis tools, we turned to FinBERT, a pre-trained NLP model specifically designed for financial texts. FinBERT, available on Hugging Face, is fine-tuned on a large corpus of financial news and reports, enabling it to accurately interpret the nuanced language of financial markets.

FinBERT excels at understanding context and domain-specific phrases, making it significantly more accurate than general-purpose models. For instance, terms like "closure" and "surge" can carry vastly different sentiments depending on the context. FinBERT can accurately analyze news headings such as "360 Energy Liability Management Accelerates Environmental Site Closure Business with Strategic Acquisition," recognizing it as a positive development. Similarly, it correctly interprets "Bitcoin's surge confuses even pro traders: 'It's trading like a Treasury'" as conveying a negative sentiment. By deciphering the nuanced meanings behind such terms, FinBERT delivers precise and actionable insights.

Additionally, FinBERT’s context awareness allows it to handle complex linguistic features such as negation and sarcasm, which are often challenging for general-purpose tools. Its training on financial-specific language ensures that it captures the subtle nuances and jargon unique to the financial domain. This capability ensures that our sentiment analysis is both accurate and relevant to the financial domain.

Here’s a code snippet demonstrating how we used FinBERT.

``` Python
from transformers import pipeline
pipe = pipeline("text-classification", model="yiyanghkust/finbert-tone")
pipe(news_article)
```

Let's consider this fictional news. The result shows that the news article input has a positive sentiment with 99.9% confidence. In contrast, TextBlob classified the same passage as neutral, failing to capture the positive sentiment evident in the article. This stark difference highlights FinBERT’s accuracy in analyzing financial news, providing more reliable and actionable insights.

> Fictional News March 18, 2025  
> FinText Limited reported a 15% decline in revenue this quarter,
> citing market volatility and slower enterprise adoption.
> On the same day, CEO Alvin Ku announced the launch of Project NLP in FINA4350.
> Despite the revenue drop, analysts at Morgan Stanley remain optimistic.
> Investors are watching closely as FinText Limited pivots toward cutting-edge text analytics advancements to fuel future growth.

``` Python
print(TextBlob(news_article).polarity)
# TextBlob Output: Neutral (-0.0357)

pipe(news_article)
# FinBERT Output: Strong Positive (0.9994)
```

## Extracting Granular Insights with Named-Entity Recognition (NER)

While sentiment analysis provides a high-level view of market sentiment, it lacks granularity. To address this, we incorporated NER using SpaCy, an open-source NLP library with fast and accurate entity recognition capabilities.

SpaCy’s deep-learning based recognition system is highly efficient, making it ideal for processing large volumes of financial text. It also allows for customization, enabling us to identify and classify domain-specific entities such as company names, stock tickers, and financial terms with high accuracy.

Here’s an example of how we used SpaCy for NER. The model successfully identified and labelled entities such as person, date, and organization, as shown in the code snippet below. This capability is particularly useful for extracting structured information from unstructured financial texts, enabling us to analyze relationships between entities and gain deeper insights into market trends and events.

``` Python
import spacy
nlp = spacy.load("en_core_web_lg")
doc = nlp(news_article)
spacy.displacy.render(doc, style="ent", jupyter=True)
```

![Picture showing the output of NER]({static}/images/group-FinText-NER_spacy.png)

By combining FinBERT with SpaCy’s NER capabilities, we linked sentiment scores to specific entities mentioned in financial news. For example, rather than assigning a general sentiment score to an article discussing multiple companies, our system identifies which companies are being discussed and associates each with its respective sentiment. This entity-specific approach ensures that investors can focus on the news most relevant to their portfolios.

## Conclusion: A Continous Exploration of Financial NLP

Our project highlights the power of combining sentiment analysis, NER, and domain-specific models like FinBERT to tackle the challenges of financial NLP. By leveraging these advanced techniques, we can extract actionable insights from financial texts, enabling more informed decision-making in the markets. As we move forward, we will continue exploring advanced methods for NLP in financial markets to refine our approach.
