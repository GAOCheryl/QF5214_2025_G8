# QF5214_Group 8 Team_2 README    

## 1 Overview
  This project implements a Natural Language Processing (NLP) pipeline used for sentiment and emotion analysis tailored for financial text data on X. The system ingests and filters data collected, extracts sentiment signals from sources and then aggregates them to support quantitative investment strategies. The pipeline consists of three core components:
  
## 2 Data Filtering
  This module provides a complete pipeline which is streamlined and fast for cleaning and filtering financial text data (e.g., stock-related posts on X) as a pre-step for NLP process.
- **Cleaning and Filtering**
  -  Lowercasing, emoji/URL/user/tag cleaning  
  -  Detection and filtering of irrelevant or promotional content  
  -  Removal of `$XXXX`-style stock ticker symbols  
  -  Filters short, meaningless, or spammy text  
- **Input Format**
  - You will need a CSV file with at least **two columns**:
    - `Text`: The raw text (e.g., post or comment on X)  
    - `Company`: The corresponding target stock symbol 
  - Example:

  | Text                                        | Company |
  |---------------------------------------------|---------|
  |✨Top stocks with TA score trending DOWN (SP500):  https://t.co/S59zN18Con  |$VRTX|
  
- **Onput**
  - The cleaned and filtered texts will be stored in a new column `'Cleaned_Text'`
 
## 3 NLP
- **3.1 Sentiment Emotion Analyzer**


  FinBERT, a transformer model pre-trained on financial text, is used to extract financial sentiment from tweet data. The model outputs probabilities for the sentiment categories positive, negative, and neutral.
  - **Model Loading**:
  
    FinBERT is loaded once via a Hugging Face pipeline. The model’s inference is wrapped in `torch.no grad()` for efficiency.
  - **Inference**:
  
    Given a cleaned tweet, FinBERT outputs probabilities for the sentiment categories.
  - **Output**:

    p(positive), p(negative), p(neutral))

 
- **3.2 Emotion Model Scores**

  This script evaluates multiple pre-trained emotion classification models on a labeled dataset. It maps model-specific labels to a universal emotion set and computes classification metrics using batch inference.
  - **Models:**
    An ensemble of three emotion models is used, weighted based on accuracy and F1 scores:
  | Model                                        | Weight |
  |----------------------------------------------|--------|
  | Model 1: `michellejieli/emotion text classifier`| Weights of 0.2|
  | Model 2: `j-hartmann/emotion-english-distilroberta-base`| Weights of 0.4|
  | Model 3: `bhadresh-savani/distilbert-base-uncased-emotion`| Weights of 0.4|
  - **Maping:**

    Each model’s output is converted to a distribution over the universal emo-tions using predefined mapping dictionaries:
    - `["sadness", "joy", "anger", "fear", "disgust", "surprise"]`
  - **Batch:**

    Batch inference using HuggingFace pipelines.


- **3.3 Aggregation and Core Method's Output Format**

  The `nlp(text)` method in `SentimentEmotionAnalyzer.py` (usage written in this file) returns a list containing all the elements below. And by `aggregate.py` we get results and their corresponding column names:
  - **FinBERT sentiment scores:**
    - FinBERT positive score :`'Positive'`
    - FinBERT negative score :`'Negative'`
    - FinBERT neutral score  :`'Neutral'`
  - **Aggregated emotion distribution.:**
    - Surprise emotion score :`'Surprise'`
    - Joy emotion score      :`'Joy'`
    - Anger emotion score    :`'Anger'`
    - Fear emotion score     :`'Fear'`
    - Sadness emotion score  :`'Sadness'`
    - Disgust emotion score  :`'Disgust'`
  - **Intent-based sentiment (trading signal):**
    - Buy, Sell or Neutral   :`'Intent Sentiment'`

## 4 Processing
- **Local Processing**: The outputs from the NLP pipeline (e.g., sentiment scores, emotion probabilities, intent labels) are stored in a dedicated SQL table. The aggregated data forms the basis for further analysis, including the construction of investment models and dashboard visualisations.
- **Real-time Processing**: Tracks last processed timestamp. Using SQL queries or Python’s pandas, the data is grouped by company and date. Upload and store in table `'sentiment_aggregated_live'` after aggregation from raw data.
- **Batch Processing**: Accelerated computing by taking full advantage of GPU performance (Proper version of package `Torch` based on GPU needed)