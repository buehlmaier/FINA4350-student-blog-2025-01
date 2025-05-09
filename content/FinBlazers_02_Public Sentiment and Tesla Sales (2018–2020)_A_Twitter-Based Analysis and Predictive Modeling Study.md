Title: Public Sentiment and Tesla Sales (2018–2020): A Twitter-Based Analysis and Predictive Modeling Study  
Date: 2025-05-03  
Category: Reflective Report
Tags: Group FinBlazers
Status: published  

# Public Sentiment and Tesla Sales (2018–2020): A Twitter-Based Analysis and Predictive Modeling Study

## 1. Introduction
Social media has emerged as a powerful medium for real-time public discourse and market sentiment analysis. Tesla, Inc., known for its innovative technologies and highly visible presence, is frequently discussed online, making it an ideal subject for sentiment-driven research and sales prediction.

This study analyzes approximately 3.27 million Twitter posts referencing Tesla from January 2018 to December 2020. We focus on understanding temporal tweet distribution, sentiment evolution, and their potential correlation with Tesla's US sales figures, ultimately developing predictive models for sales forecasting.

Our methodology integrates exploratory data analysis (EDA), sentiment analysis, and ensemble machine learning, providing insights into how social media data can inform business and investment decisions.

## 2. Data Acquisition and Preprocessing
### 2.1 Dataset Source
The primary dataset was obtained from Kaggle's "Tesla Tweets" collection, containing tweets posted from January 2018 to December 2020. The data was programmatically downloaded using the kagglehub library and processed using Python's pandas library. Additionally, we collected monthly US sales data from GoodCarBadCar for the corresponding period.

### 2.2 Preprocessing Steps
Our data preparation involved:

- Standardizing timestamps into datetime format for temporal analysis
- Creating monthly aggregations
- Filtering for English-language tweets (99.7% of the dataset)
- Cleaning text by removing noise (URLs, mentions, hashtags, and special characters)
- Preserving original data while storing cleaned versions for analysis

## 3. Initial Data Analysis
### 3.1 Dataset Composition
| Period           | January 2018 - December 2020 |
|------------------|------------------------------|
| **Metric**       | **Value**                    |
| **Total Tweets** | ~3.27 million                |
| **Verified Users** | 5% of total tweets         |
| **Language**     | Predominantly English (99.7%) |

### 3.2 Engagement Metrics
In order to assess influence and user interaction, we calculated the average engagement per tweet:

- **Retweets**: 2.35
- **Favorites**: 15.70
- **Replies**: 0.83

These values suggest moderate levels of engagement and indicate that Tesla-related content regularly captures user attention.

## 4. Sentiment Analysis
### 4.1 Methodology
We employed VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool, specifically designed for short, informal text such as tweets. VADER outputs a compound sentiment score ranging from -1 (extremely negative) to +1 (extremely positive).

Each tweet was assigned:

- **Positive**, **Neutral**, or **Negative** sentiment label based on thresholds.
- A monthly average sentiment score and sentiment distribution ratio were computed.


### 4.2 Sentiment Distribution
| Sentiment | Percentage | Description                                                      |
|-----------|------------|------------------------------------------------------------------|
| Positive  | 40%        | Reflecting optimism, support for Tesla’s technology, or stock performance. |
| Neutral   | 37%        | Informational tweets or non-opinionated commentary.              |
| Negative  | 23%        | Critical tweets about product issues, controversies.             |

The distribution demonstrates a predominantly positive sentiment in Tesla-related tweets during the analysis period. Besides, sentiment peaks often coincided with product announcements or earnings reports.

## 5. Predictive Modeling
### 5.1 Feature Engineering
We created comprehensive feature sets including:

- Historical sales data (1-3 month lag)
- Sentiment metrics (monthly averages and trends)
- Tweet volume indicators
- Temporal features (month, year)


### 5.2 Model Development
With clean data and sentiment metrics in hand, we began building models to predict Tesla sales. Taking an incremental approach, we started simple and added complexity.

#### Random Forest
Our baseline model captured basic patterns well, particularly monthly seasonality in sales. Feature importance analysis revealed sentiment trends were indeed predictive.

#### XGBoost
This gradient-boosted model improved on the Random Forest's performance, better handling complex relationships between sentiment, tweet volume, and sales.

#### LSTM Neural Network
Our first foray into deep learning used Long Short-Term Memory networks to model temporal dependencies. After struggling with data reshaping (a common LSTM challenge), we achieved decent results capturing multi-month trends.

#### Ensemble Model (Stacking)
The real breakthrough came when we combined all three approaches using stacking. A linear regression meta-learner weighted each model's predictions, yielding our most accurate forecasts and achieved best overall performance.

### 5.3 Model Performance
The ensemble model demonstrated superior performance. First, it captured both regular patterns and some anomalous behavior. It also better handled the significant sales volatility in 2020 and showed robust performance across different market conditions.

## 6. Key Findings and Discussion
### 6.1 Sentiment-Sales Relationship
A positive correlation was observed between Twitter sentiment and Tesla sales. Public sentiment trends often preceded actual sales shifts by 1 to 2 months. Sales volume surges in tweets were typically aligned with company events (e.g., Cybertruck launch, Q2 earnings), suggesting social media buzz indicates real demand shifts.

### 6.2 Predictive Insights
The models performed well on regular patterns; however, unexpected sales spikes, such as in Q3 2020, were difficult to forecast due to external factors (e.g., COVID-19 recovery, government incentives). Overall, the ensemble approach provided the most balanced predictions.

## 7. Conclusion
This study highlights the potential of combining social media sentiment analysis with machine learning for sales forecasting. For Tesla, a brand deeply interwined with public attention and discourse, Twitter sentiment served as a valuable proxy for market demand. The social media signals offer a valuable complement to traditional sales forecasting methods by capturing real-time public sentiment and engagement, which reflect emerging consumer behaviours and interests. Besides, trends in online sentiment can act as early indicators for shifts in market demand, allowing businesses to respond proactively. In our analysis, among the various predictive models tested, XGBoost emerged as the most robust and accurate, demonstrating strong performance in capturing complex patterns and maintaining reliability across different market conditions. This finding highlights the value of advanced machine learning models in utlizing unstructured social media data to inform strategic decision-making and business planning.

