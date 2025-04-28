---
Title: Tesla Sales Decoded: A Journey Through Data-Driven Narratives (by Group "FinBlazers")
Date: 2025-03-24 17:00
Category: Reflective Report
Tags: Group FinBlazers
---

## Group Members

- **Angus Fung**  
  Majoring in Financial Technology with minors in Finance and Computer Science. Passionate about entrepreneurship, business strategy, and emerging tech. Loves problem-solving, exploring new ideas, and playing basketball and table tennis.  [GitHub](https://github.com/angusf777)

- **Haoze Lu (Peter)**  
  Majoring in Applied AI with minors in Finance and Computer Science. Passionate about AI technology, financial technology, and business strategy analysis. Loves travelling, badminton, and swimming. Always exploring.

- **Leung Cheuk Yiu (Janice)**  
  Majoring in Financial Technology with a keen interest in digital experience transformation, digital strategy, and business analysis. Passionate about utilizing technology to enhance user experiences and drive business success.

- **Hui Jing Tung (Bernice)**  
  Majoring in Financial Technology. Interested in leveraging technology in the financial world, particularly in streamlining processes and driving product insights in the banking field.

- **Qian Yongkun Jonathan**
  Majoring in Financial Technology with minors in Finance and Computer Science. Passionate about the integration of cutting-edge technology in the finance industry.  Loves playing football and music.


## The Spark: Igniting Our Journey to Decode Tesla's Sales Narrative

In this blog post, we will illustrate how the dynamics of Tesla’s market sparked our idea to predict Tesla EV sales using natural language processing (NLP). We’ll cover our journey from the initial concept to the concrete steps we’re taking. Along the way, we’ll outline our motivations, discuss the challenges we anticipate, and describe the approach we’ve adopted to refine our methods and address potential obstacles.

Our goal is to provide a transparent overview of our process, highlighting the ways in which we have tested and validated our hypotheses. By continuously updating our progress, we hope to offer insights into our problem-solving journey, showcase our team’s dynamic collaboration, and ultimately demonstrate how qualitative narratives can enhance quantitative forecasting. Stay tuned as we share our milestones, learning experiences, and ongoing refinements to achieve our end goals.

## Tesla's Odyssey: From Revolutionary Beginnings to Turbulent Times

Our idea originated from some spirited conversations among our team, and it became crystal clear that Tesla's story is about so much more than just the numbers on a balance sheet. It is about the compelling narratives that have shaped its path.

Since its founding, Tesla has revolutionized the automotive and energy sectors, shaking up traditional norms with its groundbreaking technology and bold vision. However, this journey has had its fair share of bumps along the way. From fierce market competition to production challenges, Tesla has navigated a landscape filled with obstacles. 

Today, Tesla faces increasing pressure from domestic EV manufacturers, particularly in China, its most important market. In 2024 alone, Tesla sold over 650,000 cars in China, yet its market share is shrinking. The competition has intensified, with companies like BYD aggressively pricing their vehicles, leading to a price war that is eroding Tesla’s profit margins. In January 2025, Tesla’s sales in China dropped 11.5% year-over-year, while BYD surged 47% in the same period.

The challenges extend beyond China. In Europe, Tesla’s sales declined 45% year-over-year in January 2025, even as overall EV adoption in the region continued to grow. Contributing factors include rising competition, the reduction of government subsidies, and—perhaps most controversially—Elon Musk’s political activism, which has influenced public sentiment.

Financially, Tesla is also seeing signs of strain. While it once boasted industry-leading profit margins, recent price cuts to maintain competitiveness have weighed heavily on profitability. In Q4 2024, Tesla’s revenue grew by only 2%, while operating income fell 23% due to rising costs. These pressures highlight the need for a deeper analysis of how external narratives—ranging from competition to investor sentiment—impact Tesla’s EV sales trajectory.

Yet, through all these turbulent times, Tesla's commitment to innovation and sustainability continues to inspire. Despite financial pressures and market challenges, Tesla remains at the forefront of technological advancement, for example, it has been investing heavily in artificial intelligence and autonomous driving. The company is training its Full Self-Driving (FSD) software with billions of miles of real-world driving data, pushing the boundaries of AI-driven autonomy. While its AI-driven autonomy ambitions are promising, the monetization timeline remains uncertain. Investors are awaiting concrete progress on Tesla's Robotaxi initiative, which could be a game-changer to sales if successful but remains speculative at this stage.

Our project aims to capture these multifaceted narratives by utilizing NLP to analyze how factors such as the launch of a new model, competitive pressures, and shifts in investor sentiment impact Tesla’s electric vehicle (EV) sales. By doing so, we hope to bridge the gap between qualitative narratives and quantitative performance, providing a fresh perspective on how revolutionary ideas can both drive and challenge market success.

## A Tiered Approach? Why?

Taking on a project like predicting Tesla EV sales with natural language processing (NLP) can be quite daunting. The vast amount of data, complexities of modern NLP techniques, and the rapidly evolving market conditions can be easily overwhelming for us. We understand that if we jump headfirst into creating a fully integrated deep-learning model right away, we could face significant challenges. For example, we might encounter delays in data collection, overfitting or convergence issues, and unexpected gaps in our data could affect our predictions. Trying to manage all these factors simultaneously could jeopardize our ambitious goals. 

## Expected Challenges and Possible Failures

First of all, the data quality and availability present significant challenges for our project. Inconsistent or incomplete datasets from sources like social media, news, and financial filings might hinder our model training efforts, making it difficult to draw accurate insights. Moreover, the complexity of NLP models introduces additional difficulties; advanced techniques such as transformers and deep topic modeling may lead to overfitting if not managed carefully. In addition to these technical challenges, we must also consider temporal and contextual variability. The factors driving Tesla's sales can shift rapidly due to sudden policy changes, competitive moves, or fluctuations in public sentiment, which makes it tough to maintain stable model performance over time. Furthermore, as students, we frequently face resource constraints, including limited computational power and time. This complicates our ability to handle large-scale data processing or conduct deep learning experiments effectively. These factors create a challenging landscape for our analysis, requiring careful planning and strategic approaches to overcome.

## Breaking It Down Into Tiers 

We’ve structured our approach into clear, incremental tiers to mitigate these risks and ensure steady progress. This tiered framework allows us to validate foundational ideas in Tier 1 before scaling up to more sophisticated Tier 2 and 3 methods. Should we encounter insurmountable challenges—be it data quality issues, model complexity, or time constraints—our early wins (such as basic sentiment analysis) will still provide valuable insights, ensuring that we don’t walk away empty-handed. This incremental methodology also offers natural checkpoints for refining our approach, adjusting scope, and re-aligning with data and resource availability realities. Ultimately, the tiered strategy helps us stay agile, systematically building toward our most ambitious goals without risking complete project failure if one element proves too difficult.

### Tier 1 – Foundational Proof-of-Concept
- **Goal:** Establish a baseline correlation between simple sentiment metrics and Tesla sales.
- **Data:** Focus on a single source (e.g., news headlines) plus Elon Musk's Tesla-related tweets.
- **Methods:** Basic sentiment analysis (VADER or TextBlob), followed by simple regression to measure the relationship with monthly or quarterly sales data.

### Tier 2 – Enhanced Multi-Source Analysis
- **Goal:** Integrate additional textual inputs and employ more sophisticated NLP techniques to enrich our predictive features.
- **Data:** Multiple news outlets, Reddit forums, Tesla’s financial filings, and potentially competitor announcements.
- **Methods:** BERT-based sentiment analysis, named entity recognition (NER) for key events, and topic modeling (LDA) to capture broader themes.

### Tier 3 – Advanced Thematic Exploration
- **Goal:** Conduct deep thematic analysis to identify which narratives (e.g., competition, political controversies, new model releases) most strongly correlate with Tesla's sales.
- **Data:** All previous sources plus macroeconomic indicators, government policy documents, and further competitor data.
- **Methods:** Apply advanced LDA or other topic modelling techniques to quantify thematic trends and correlate them with sales, exploring time-lagged effects to better understand causal relationships.

## Data Sources

So far, we have identified key datasets to support our analysis:

- **Tesla Sales Data & Reports:**  
  Detailed monthly sales data for Tesla and other major car manufacturers from 2015 to 2024.  
  [Tesla Sales Data & Reports | GCBC](https://www.goodcarbadcar.net/brands/tesla-sales-data-reports/)

- **Twitter Dataset:**  
  10,000 Twitter posts about Tesla, with detailed post dates up to Dec 7, 2022.  
  [hugginglearners/twitter-dataset-tesla on Hugging Face](https://huggingface.co/datasets/hugginglearners/twitter-dataset-tesla)  
  The code we use is as follows:
   
          curl -X GET "https://datasets-server.huggingface.co/rows?dataset=hugginglearners%2Ftwitter-dataset-tesla&config=default&split=train&offset=0&length=100"


- **Financial News Articles (2020-2024):**
  Financial news about Tesla, primarily without specific dates.
  [Microsoft Tesla Finance News Articles (2020-2024) on Kaggle](https://www.kaggle.com/datasets/journeyyouyeonkim/microsoft-tesla-finance-news-articles2020-2024)

- **Financial News on Tesla and Elon Musk (2022):**
  Articles focused on Tesla and Elon Musk in 2022. 
  [Newspapers Dataset on Kaggle](https://www.kaggle.com/datasets/saleepshrestha/newspapers)   
  The code we use is as follows:
   
          import kagglehub    
          # Download latest version   
          path = kagglehub.dataset_download("saleepshrestha/newspapers")   
          print("Path to dataset files:", path)   


## Analysis Plan

Looking at all the datasets we have obtained at hand, our first step into the project may be making use of the temporal information in the Twitter posts, and picking that financial news with a clear date or is known to belong to 2022, to use Dec 2022 as a cutting point, and use the dataset 1-2 years before Dec 2022 to predict the sales of the following year (2023) and validate with the existing figures. Apart from that, we also have an accessible dataset about the sales of competitors of Tesla, the correlation of sales with competitors is an achievable next step. In the meantime, we will be continuing with collecting new datasets to have a better analysis in each Tier.

## Conclusion

Through our tiered approach, we look forward to harnessing the power of NLP to dissect the intricate narratives that drive Tesla's sales. By breaking down our ambitious goal into manageable, incremental steps, we aim to progressively build a robust model that accurately captures and predicts market dynamics. We are currently acquiring data sources and preprocessing them to fit our analytical needs. Stay tuned for further updates and detailed insights in our next post.
