---
Title: Evaluating LLMs as an Alternative for VADER and FinBERT in Financial Sentiment Analysis (by Group "FinText")
Date: 2025-04-28 00:00
Category: Reflective Report
Tags: Group FinText
---

## Introduction

Sentiment analysis has become an essential tool in trading, extracting market-moving signals from data sources like news articles. Our News-driven NLP Quant project currently employs two established methods: VADER, a lexicon-based approach, and FinBERT, a financial domain-specific model. With the emergence of sophisticated large language models (LLMs), we must examine whether these advanced systems can surpass traditional techniques in financial sentiment analysis.

This comparative study analyses three real-world financial news cases to assess the strengths and limitations of each approach. The findings provide actionable insights for practitioners considering sentiment analysis methodologies. For this analysis, we utilised DeepSeek as our test LLM.

## Comparative Case Analysis

| Approach    |                | Case 1   | Case 2   | Case 3   |
|-------------|----------------|----------|----------|----------|
| **VADER**   | Title sentiment    | 0        | 0.1531   | -0.4938  |
|             | Summary sentiment  | -0.5106  | -0.2960  | 0.3818   |
| **FinBERT** | Title sentiment    | 0.9480   | -1       | -1       |
|             | Summary sentiment  | 1        | -0.9954  | -1       |
| **DeepSeek**| Title sentiment    | 0.35     | -0.65    | -0.70    |
|             | Summary sentiment  | 0.60     | -0.55    | -0.45    |

### Case 1: Market Reaction to Geopolitical News

The article **"Wall Street Rallies as West Hits Russia with New Sanctions"** presented an interesting test case for sentiment analysis. VADER produced a neutral score (0.00) for the title and negative (-0.51) for the summary, failing to capture the market's positive interpretation of events. FinBERT correctly identified positive sentiment with high confidence, while DeepSeek produced scores of +0.35 for the title and +0.60 for the summary. All three methods captured different aspects of the complex sentiment landscape.

### Case 2: Macroeconomic Downturn Report

Analysis of **"Weak Manufacturing Drags Down Q3 GDP Growth"** revealed important differences in approach. VADER produced an unexpected positive score (+0.15) for the negative-leaning headline. FinBERT generated a strongly negative classification, while DeepSeek produced scores of -0.65 for the title and -0.55 for the summary. This case highlights how different methodologies handle gradations in negative sentiment.

### Case 3: Corporate Earnings Announcement

The Zoom earnings report demonstrated the challenges of interpreting mixed financial signals. VADER produced contradictory scores (-0.49 for the title versus +0.38 for the summary). FinBERT consistently classified both components as negative, while DeepSeek provided scores of -0.70 for the title and -0.45 for the summary. This suggests that while transformer-based models may be more consistent than lexicon approaches, they can differ significantly in their interpretation of ambiguous cases.

## Methodological Comparison

### Accuracy and Nuance

FinBERT and DeepSeek captured contextual clues that VADER missed. The LLM tended to provide graduated scores (e.g., –0.55 instead of a hard –1), which can be valuable for position sizing.

### Computational Considerations

FinBERT strikes a favorable balance between speed and accuracy, processing documents within milliseconds while maintaining strong domain-specific relevance. VADER offers the fastest processing without requiring a GPU, but at the cost of reduced accuracy. LLMs, while highly capable in this task, typically demand significantly greater computational resources and project budget.

### Interpretability

FinBERT is open-source, allowing inspection of its token-level logits. In contrast, closed-weight LLMs offer limited transparency. However, both FinBERT and LLMs remain largely black-box models compared to the rule-based and inherently interpretable VADER.

## Practical Recommendations

For research baselines and prototyping, FinBERT offers a robust solution, delivering solid accuracy with modest computational overhead. When time and budget permit, with the need for fine-grained sentiment insights, LLMs provide powerful capabilities for deeper analysis. In some cases, a hybrid cascade approach is effective where FinBERT handle the first pass and escalating only complex or ambiguous cases to an LLM for finer interpretation. VADER, while limited in financial nuance, remains useful for rapid prototyping or preliminary checks, especially when GPU resources are unavailable.

## Conclusion

This comparative analysis demonstrates that both LLMs and fine-tuned, smaller models like FinBERT have distinct roles in financial sentiment analysis. While LLMs, as evidenced by articles like Fatouros et al. (2023), excel at processing nuanced financial texts and show superior correlation with market reactions, FinBERT remains indispensable when budget constraints and computational efficiency are critical.

Future work should focus on developing hybrid approaches that leverage the strengths of both specialised financial models and general-purpose LLMs, along with standardised evaluation frameworks for comparing different sentiment analysis methodologies. The financial NLP community would benefit from continued research into more efficient LLM architectures and a better understanding of how these different approaches complement each other in real-world applications.

## References

Fatouros, G., Soldatos, J., Kouroumali, K., Makridis, G., & Kyriazis, D. (2023)  
Transforming sentiment analysis in the financial domain with ChatGPT.  
Machine Learning with Applications, 12, 100508. [https://doi.org/10.1016/j.mlwa.2023.100508](https://doi.org/10.1016/j.mlwa.2023.100508)
