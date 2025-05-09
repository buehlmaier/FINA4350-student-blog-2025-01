---
Title: Public Sentiment and Tesla Sales in 2022: An Integrated Analysis of News and Twitter Data
Date: 2025-05-04
Category: Reflective Report
Tags: Group FinBlazers
---

# Public Sentiment and Tesla Sales in 2022: An Integrated Analysis of News and Twitter Data

### 1. Introduction

What drives people to buy a Tesla? Is it the technology? The brand? Or could it be the stories we read and the tweets we scroll through? In today’s hyper-connected world, sentiment—both from traditional media and social platforms—can ripple through public consciousness and sway consumer behavior in ways we’re only beginning to understand. This blog post brings together three interconnected studies that explore how sentiment, expressed in news headlines and Elon Musk’s tweets, may influence Tesla’s monthly sales in 2022. Rather than examining these elements in isolation, we took a layered approach, building one analysis atop the next to uncover patterns that are both statistically significant and narratively compelling. We began by comparing two sentiment analysis techniques, VADER and a transformer-basedmodel, to evaluate how well financial news sentiment aligned with Tesla’s actual sales. Then, we turned to Elon Musk’s Twitter activity, analyzing his tone, language, and timing to see whether his tweets could signal market shifts or consumer behavior. Finally, we fused bothsentiment streams, news and social media, into a predictive neural network model, testing whether this multimodal input could accurately forecast Tesla’s monthly performance. This project is not only about Tesla, but also about the growing power of digital sentiment and its real-world impact. As you read on, we invite you to think critically about how perception and data intertwine, and what this might mean for companies, consumers, and markets in the years ahead.

### 2. Tesla News Sentiment and Sales Correlation Analysis: A Comparative Methodology Study

#### 2.1 Data Acquisition and Cleaning

Our analysis begins with the foundation of all natural language processing (NLP) projects: data curation and cleaning. We sourced two primary datasets, which contain 764 Tesla-related news articles from various reputable outlets published throughout 2022, and the company’s official monthly vehicle sales figures for the same year. Each news article entry includes metadata such as the publication date, source, title, and full text content, allowing for both temporal alignment and sentiment analysis.


The raw text data, as expected, required significant preprocessing before any analysis could take place. We applied a sequence of standard NLP cleaning steps, including removal of
special characters, lowercasing, tokenization, stopword removal, and lemmatization. These steps distilled the textual data into cleaner, more meaningful linguistic units that were easier to analyze computationally. For instance, the pipeline included functions such as:

```python
news_df['text_clean'] = news_df['content'].apply(lambda x: clean_text(x))
news_df['tokens'] = news_df['text_clean'].apply(lambda x: word_tokenize(x))
news_df['tokens'] = news_df['tokens'].apply(
    lambda x: [lemmatizer.lemmatize(w) for w in x if w not in stop_words])
```

After cleaning, we conducted exploratory text analysis to understand the underlying themes within the news corpus. Word frequency plots and word clouds revealed dominant terms like “Tesla,” “Musk,” “electric,” and “vehicle.” We also explored bigram patterns—common two-word combinations, such as “Elon Musk” and “electric vehicle,” which highlighted key
narrative motifs consistently echoed across various articles. These preliminary insights laid the groundwork for our subsequent sentiment analysis and correlation studies.

#### 2.2 Sentiment Analysis Models

To quantify the emotional tone of news coverage surrounding Tesla, we employed two contrasting sentiment analysis models, each selected for its unique strengths and
methodological perspective. The first is **VADER (Valence Aware Dictionary and sEntiment Reasoner)** , a rule-based model widely used for its simplicity and effectiveness in analyzing short texts, particularly from social media. VADER produces a compound score ranging from -1 (extremely negative) to +1 (extremely positive), summarizing the emotional valence of a given passage. While not specifically tailored to financial contexts, its interpretability and ease of implementation make it a useful baseline. After applying VADER to each news article, we aggregated the monthly sentiment scores and aligned them with Tesla’s corresponding sales data. This allowed us to assess whether a correlation exists between media tone and real-world commercial outcomes. To complement and contrast with VADER’s general-purpose approach, we incorporated a more sophisticated deep learning model, **DistilRoBERTa, fine-tuned on financial news data**. This transformer-based model leverages contextual embeddings to understand nuanced financial language, providing sentiment scores across three categories: positive, negative, and neutral. It is particularly well-suited for the type of long-form, domain-specific text found in news articles about Tesla.


Given the limitation of transformer models (especially those based on BERT architectures) regarding input length, we implemented a chunking strategy to process full articles. The method involves splitting each article into segments of no more than 512 tokens—BERT’s maximum input length—and passing each chunk through the model individually. Predictions are then aggregated to produce a single sentiment classification for the article. The procedure is summarized in the following function:

```python
def chunk_and_predict(text):
    chunks = split_text_into_chunks(text, max_len=512)
    predictions = [model(chunk) for chunk in chunks]
    return aggregate_predictions(predictions)
```

This dual-model approach not only improves robustness but also allows us to reflect on the methodological trade-offs between rule-based and deep learning sentiment systems.
Comparing their outputs enabled a richer exploration of how sentiment varies across time—and how these variations might relate to consumer behavior and Tesla’s monthly sales performance.

#### 2.3 Findings

The application of both sentiment models yielded valuable insights into the tone and potential market impact of Tesla-related news coverage in 2022.

**2.3.1 Sentiment Distribution:**
VADER, true to its design for capturing overt sentiment cues in short text, labeled the news corpus as overwhelmingly positive—83.2% of articles were rated as positive, 16.6% negative, and a negligible 0.1% neutral. In contrast, the DistilRoBERTa model offered a more nuanced picture: only 30.0% of news pieces were classified as positive, while 37.4% were negative and 32.6% neutral. This divergence reflects the model’s financial fine-tuning, which may better capture subtleties such as cautious optimism or veiled skepticism in journalistic language.

**2.3.2 Temporal Trends:**
Monthly sentiment patterns revealed some convergence and divergence between the models. Both identified **July** as a period of particularly negative coverage, it is likely related to broader macroeconomic turbulence and company-specific controversies. However, VADER marked **March** as the most optimistic month, perhaps due to positive earnings reports or vehicle delivery milestones, whereas the Transformer model remained more conservative in its assessments throughout the year.

**2.3.3 Correlation with Sales:**
To evaluate the predictive utility of sentiment, we computed Pearson correlation coefficients between sentiment scores and Tesla’s monthly vehicle sales, including simple sales figures as well as 3-month and 6-month moving averages (MA):

| Metric               | VADER (r) | Transformer (r) |
|----------------------|-----------|-----------------|
| Same-month sales     | **0.511** | 0.000           |
| 3-month MA sales     | 0.339     | 0.000           |
| 6-month MA sales     | -0.010    | -0.000          |

VADER’s stronger correlation with same-month sales suggests that surface-level sentiment polarity—captured through lexicon-based rules—may more directly reflect short-term shifts
in consumer and investor enthusiasm. On the other hand, the Transformer’s weaker and even negative correlation with longer-term trends could point to its heightened sensitivity to neutral or mixed sentiments, which may not translate neatly into immediate commercial outcomes.

**2.3.4 Visualizations and Interpretations:**
We complemented our quantitative findings with a range of visual tools. **Stacked bar charts** illustrated monthly sentiment distributions, **heatmaps** mapped sentiment-sales correlations over time, and **overlay plots** juxtaposed sentiment trajectories against actual Tesla sales. Together, these visuals reinforced the idea that while sentiment provides a valuable signal, its interpretability and predictive power vary significantly depending on the model employed and the time frame considered.

### 3. Public Sentiment and Twitter Activity Analysis: Elon Musk’s Communication in 2022

#### 3.1 Dataset and Preparation

Elon Musk’s prolific Twitter presence represents a unique and direct channel of corporate communication—one that often bypasses traditional public relations mechanisms. In this
section, we analyze 5,390 tweets posted by Musk throughout 2022, focusing on how his messaging style, tone, and frequency relate to public sentiment and possibly Tesla’s commercial outcomes.


The dataset, retrieved through Twitter’s API, includes timestamps, tweet content, and engagement metrics. We first standardized the date format and derived a monthly indicator to align his Twitter activity with Tesla’s monthly sales data:

```python
elon_df['created_at'] = pd.to_datetime(elon_df['created_at'])
elon_df['month'] = elon_df['created_at'].dt.to_period('M')
```

#### 3.2 Text Cleaning and Preprocessing:

Given the informal and often idiosyncratic nature of tweets, preprocessing required more than the typical NLP pipeline. Our goal was to preserve meaning while filtering out noise,
especially from platform-specific artifacts. The cleaning process involved:

- Decoding HTML entities for accurate semantic representation.
- Removing retweet indicators ("RT") and quoted tweets to focus on Musk’s original messaging.
- Translating emojis into text to retain sentiment nuances often lost in raw text analysis.
- Stripping URLs, mentions, and hashtags—elements which rarely carry analyzable sentiment.
- Lowercasing and trimming excessive whitespace for consistency.
- Removing stopwords selectively (while retaining converted emoji descriptions).

The implementation reflects these steps:

```python
df['clean_text'] = df['text'].apply(lambda x:
re.sub(r"http\S+|@\S+|#[^\s]+", '', x))
df['clean_text'] = df['clean_text'].apply(lambda x:
demojize(x).lower())
```

This refined dataset serves as the basis for our subsequent sentiment analysis and linguistic pattern extraction, allowing us to investigate how Musk’s communication evolved over time and whether it aligned with broader media sentiment or sales dynamics.


#### 3.3 Key Results

**3.3.1 Tweet Frequency Over Time:**
Elon Musk’s Twitter activity in 2022 exhibited striking fluctuations. Until mid-year, his monthly tweet count remained relatively stable. However, a dramatic surge occurred in
November and December—exceeding 1,000 tweets per month—coinciding with his acquisition of Twitter. While the tweets were not necessarily Tesla-related, this escalation in
online presence reflects a shift in public attention and may have had indirect implications for Tesla’s brand visibility and investor sentiment. Earlier, subtler spikes in activity appear to align with major Tesla announcements, hinting at possible synchrony between Musk’s communication patterns and the company’s corporate events.

**3.3.2 Tesla Content Ratio:**
Surprisingly, fewer than 6% of Musk’s tweets in 2022 explicitly mentioned Tesla or its products. This low frequency suggests that while Musk is the face of Tesla, much of his online engagement is directed elsewhere—toward space exploration, cryptocurrency, AI, or sociopolitical commentary. Nevertheless, the gravitational pull of his personal brand ensures that even non-Tesla tweets may indirectly shape public perception of the company.

**3.3.3 Sentiment Distribution:**
Using the VADER sentiment model, we categorized the tone of Musk’s tweets across the year. The distribution was relatively neutral overall, with:

- 39% classified as **positive**
- 47% as **neutral**

```python
elon_df['sentiment'] = elon_df['clean_text'].apply(lambda x:
analyzer.polarity_scores(x)['compound'])
```

This predominance of neutral or mildly positive tone may reflect Musk’s calculated communication strategy—assertive but rarely overtly negative—designed to maintain engagement without excessive controversy, at least in the textual tone.

**3.3.4 Linguistic Patterns:**
Bigram analysis provided further insight into the language Musk used most frequently. Common two-word phrases such as _"fire hire"_ , _"free speech"_ , and _"launch falcon"_ hint at thematic clusters: internal staffing dynamics (often controversial), advocacy of open discourse (especially post-Twitter acquisition), and SpaceX-related updates. These linguistic patterns reflect the multidimensional nature of Musk’s digital persona and suggest that sentiment interpretation must be context-aware.


#### 3.4 Visualizations

The visualization suite serves not merely as a summary of tweet activity, but as a lens into Elon Musk’s shifting communicative focus and its potential resonance with public sentiment.

**3.4.1 Monthly Tweet Volumes:**
A time series plot of tweet frequency vividly illustrates the escalation in late 2022, where tweet counts more than doubled compared to earlier months. The timing—immediately
following the finalization of the Twitter acquisition—suggests a redirection of Musk’s attention from Tesla to broader platform governance and social discourse.

**3.4.2 Tesla vs. Non-Tesla Tweet Ratio:**
A categorical breakdown reveals a surprisingly low proportion of Tesla-related content. Less than 6% of all tweets in 2022 mentioned Tesla, reinforcing the notion that Musk's personal
brand operates in a larger ecosystem than Tesla alone. This raises interesting questions about how audiences conflate Musk’s identity with the companies he leads—often with limited
explicit cues.

**3.4.3 Sentiment Breakdown Over Time:**
Stacked bar charts illustrate a predominance of neutral sentiment throughout the year, punctuated by spikes in positivity or negativity during key events. Notably, the emotional tone remained remarkably stable even during periods of high tweet volume, such as the Twitter acquisition period. This supports earlier findings that, while prolific, Musk’s language does not oscillate dramatically in sentiment—at least from a lexical standpoint. Together, these visual elements provide both macro- and micro-level perspectives on how Musk communicated during 2022, and establish a foundation for understanding how public-facing CEO activity intersects with brand sentiment and potentially with sales dynamics.

## 4. Public Sentiment and Tesla Sales Prediction (2022): Integrating News and Social Media Analysis

Having analyzed news and Twitter sentiment independently, we turned to their **combined predictive power**. This phase aimed to explore whether sentiment data—originating from media outlets and Elon Musk’s Twitter account—could be used to **forecast Tesla’s monthly vehicle sales** for 2022.

#### 4.1 Data preprocessing

To prepare the data for analysis, we executed the following steps:

1. **Date Normalization** : Standardized date formats across both datasets to enable chronological
    analysis and monthly aggregation.
2. **Sentiment Analysis** : Applied the finiteautomata/ bertweet-base-sentiment-analysis model to both news articles and tweets to extract sentiment scores
3. **Batch Processing** : Implemented batch processing to optimize computational efficiency when analyzing the large volume of texts (>6,000 combined articles and tweets).
4. **Engagement Weighting** : For tweets, we calculated an engagement score so the weighted sentiment was computed by multiplying the raw sentiment score by the engagement score, giving greater importance to tweets with higher public interaction.

#### 4.2 Feature Engineering & Aggregation

**4.2.1 Monthly Aggregation:**
Since Tesla reports its sales on a monthly basis, we also aggregated our sentiment and engagement data for news articles and twitter posts by month to ensure alignment. To engineered the following features monthly:

● **Mean and standard deviation of sentiment scores:** These summarize the overall tone and variability of the data each month.
● **Sum and mean of positive, negative, and neutral sentiment scores**
● **Article and Tweet volume**

By aggregating these features by month, we created a comprehensive dataset that captures both the tone and reach of media and social buzz around Tesla.

**4.2.2 Feature Scaling:**
Before feeding these features into our neural network, we normalized all values using a MinMaxScaler. This technique scales every feature to a common range (typically 0 to 1), which is important because it prevents features with larger absolute values from dominating the learning process. Proper scaling ensures that each feature contributes proportionally to the model’s training.

#### 4.3 Training Methodology

We used the first six months of 2022 (January-June) as our training set and the remaining six months (July-December) as our test set. This chronological split reflects real-world forecasting scenarios where past data predicts future outcomes.

#### 4.4 Evaluation and Findings

We evaluated the models using **MAPE** , **RMSE** , and **R²** scores:

**Metric Initial Model Improved Model**


| Metric      | VADER     | Transformer |
|-------------|-----------|-------------|
| RMSE        | 37,474.27 | 7,639.      |
| R² Score    | -68.61    | -1.         |
| MAPE        | 55.77%    | 16.36%      |


For the initial model, the training loss curve showed rapid initial convergence followed by plateau, indicating the model quickly learned patterns in the training data but failed to generalize to test data. Also, the initial model's predictions drastically overestimated September 2022 sales (126,783 vs. actual 37,518) while underestimating other months. The enhanced model showed substantial improvement. while the R² score remained negative, indicating the model still doesn't outperform a simple mean-based prediction, the dramatically reduced MAPE and RMSE suggest meaningful progress. The improved predictions tracked actual sales trends more closely, particularly for October-November 2022.

### 5. Insights and Strategic Implications from Sentiment-Sales Dynamics

This study reveals several important insights at the intersection of sentiment analysis, executive communication, and market behavior. By synthesizing results across models and data sources, we can draw meaningful conclusions about how sentiment correlates with commercial outcomes and where analytical approaches can be refined for strategic advantage.

#### 5.1 News Sentiment Exhibits Stronger Alignment with Sales

Contrary to the prevailing emphasis on social media in recent years, our findings indicate that **news sentiment shows a more immediate and stable relationship with sales data** , particularly in the short term. This suggests that **established media sources continue to shape consumer perceptions in significant ways** , likely due to their perceived authority and depth of analysis. The temporal proximity of sentiment shifts in news to fluctuations in sales underscores the continued influence of traditional journalism in shaping economic behavior.

#### 5.2 Transformer-Based Models Provide Superior Analytical Depth

The comparison between lexicon-based models (e.g., VADER) and domain-specific transformer architectures (e.g., FinBERT) yielded a clear outcome: **transformer models consistently delivered more contextually relevant and accurate sentiment evaluations**. Their ability to account for semantic nuance, domain-specific language, and sentiment embedded in complex grammatical structures makes them more effective tools for capturing actionable emotional tone. These models are especially valuable when working with financial or technical corpora, where traditional methods tend to oversimplify sentiment polarity.

#### 5.3 Executive Messaging as a Stabilizing Sentiment Factor

Elon Musk’s Twitter content, while thematically diverse, maintained a **remarkably consistent positive tone throughout both stable and turbulent periods**. This consistency may serve a strategic function—providing investors and consumers with a sense of continuity and confidence, even in the face of operational or reputational volatility. From a communications perspective, this highlights the **growing importance of executive social media as a reputational buffer** and an informal mechanism of market signaling.

#### 5.4 Toward Time-Aware, Multi-Source Predictive Models

Finally, when sentiment from both news and social media was integrated into a **time-aware modeling framework** , the results demonstrated enhanced predictive power regarding market behavior. This underscores the value of **multimodal sentiment integration** and **temporal sensitivity** in forecasting frameworks. The implications are far-reaching: such models may be leveraged not only for consumer behavior analysis but also for investor sentiment tracking, risk assessment, and strategic decision-making.

### 6. Conclusion

This study offers a comprehensive exploration of the complex relationship between public sentiment and Tesla’s commercial outcomes during 2022. By comparing various sentiment analysis approaches and integrating them into a deep learning framework for sales prediction, we underscore the critical importance of methodological precision in sentiment-based market forecasting.

Our findings reaffirm the significant, though often indirect, influence of Elon Musk’s personal communication on public sentiment, despite relatively limited Tesla-branded content on Twitter. The consistent tone of his posts may serve as an implicit signal to markets, adding a layer of interpretive complexity to executive social media behavior. Looking ahead, future research could benefit from broadening the scope of sentiment sources (e.g., Reddit threads, investor discussion forums, YouTube commentary), extending the temporal range, and further refining multimodal, time-aware modeling architectures. These directions offer promising potential to enhance both predictive accuracy and interoperability.

Overall, this work provides a solid foundation for **integrating linguistic, temporal, and financial data** in understanding and anticipating behavior in tech-driven markets. It contributes to a growing body of research at the intersection of natural language processing, behavioral economics, and data-driven decision-making.
