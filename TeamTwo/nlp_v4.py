import re
import math
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class SentimentEmotionAnalyzer:
    """
    This class provides end-to-end analysis of text, including:
      1) Loading all NLP pipelines (FinBERT + 3 emotion models) only once (in __init__),
      2) Tokenising and merging numeric tokens with '%',
      3) Classifying intent (Buy/Sell/Hold/Neutral/Spam),
      4) Aggregating emotions from multiple models into a universal set,
      5) Computing overall emotion confidence via inter-model agreement,
      6) Printing results (financial sentiment, emotion distribution, confidence, intent).
    """

    def __init__(self, k_intent=0.5):
        """
        Step 1: Initialise and load all resources.

        - Load spaCy for tokenisation.
        - Set an intent confidence scaling constant (k_intent).
        - Define intent keywords and universal emotion sets with mapping dictionaries.
        - Load each pipeline once (FinBERT for financial sentiment, plus 3 emotion models),
          so they won't be reloaded on every analysis call.
        """
        # 1.1) Load spaCy
        self.spacy = spacy.load("en_core_web_sm")

        # 1.2) Store scaling constant for intent classification
        self.k_intent = k_intent

        # 1.3) Define intent keywords
        self.intent_keywords = {
            "Buy": [
                "buy", "accumulate", "add", "long", "purchase", "invest", "acquire", "win", "dip",
                "snap", "enter", "load", "bet", "pick", "collect", "capitalise", "dca", "moon"
            ],
            "Sell": [
                "sell", "dump", "short", "liquidate", "dispose", "offload", "unload", "low",
                "exit", "cash", "divest", "cut", "flip", "unwind", "loss", "lose", "peak"
            ],
            "Hold": [
                "hold", "keep", "retain", "maintain", "stay", "remain", "observe"
            ],
            "Neutral": [
                "question", "news", "update", "inquire", "info", "ask", "rumor",
                "hearing", "report", "opinion", "view", "insight", "thoughts", "feedback",
                "query", "inquiry"
            ],
            "Spam": [
                "promotion", "advertisement", "sponsored", "promo", "ad", "click",
                "paid", "marketing", "endorsement", "commercial", "sponsor", "brand", "partnership",
                "subscribe", "follow"
            ]
        }

        # 1.4) Define universal emotion set and mappings
        self.universal_emotions = ["surprise", "joy", "anger", "fear", "sadness", "disgust"]
        self.mapping_model1 = {
            "surprise": "surprise",
            "joy": "joy",
            "anger": "anger",
            "fear": "fear",
            "sadness": "sadness",
            "disgust": "disgust"
        }
        self.mapping_model2 = {
            "joy": "joy",
            "surprise": "surprise",
            "fear": "fear",
            "anger": "anger",
            "disgust": "disgust",
            "sadness": "sadness"
        }
        self.mapping_model3 = {
            "surprise": "surprise",
            "joy": "joy",
            "anger": "anger",
            "fear": "fear",
            "sadness": "sadness",
            "disgust": "disgust"
        }

        # 1.5) Load pipelines once
        # (A) FinBERT
        finbert_model_name = "ProsusAI/finbert"
        finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
        finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name, num_labels=3)
        self.finbert_pipeline = pipeline(
            "text-classification",
            model=finbert_model,
            tokenizer=finbert_tokenizer,
            top_k=None
        )

        # (B) Model 1: michellejieli/emotion_text_classifier
        model_name1 = "michellejieli/emotion_text_classifier"
        tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
        model1 = AutoModelForSequenceClassification.from_pretrained(model_name1)
        self.michellejieli_pipeline = pipeline(
            "text-classification",
            model=model1,
            tokenizer=tokenizer1,
            top_k=None
        )

        # (C) Model 2: j-hartmann/emotion-english-distilroberta-base
        model_name3 = "j-hartmann/emotion-english-distilroberta-base"
        tokenizer3 = AutoTokenizer.from_pretrained(model_name3)
        model3 = AutoModelForSequenceClassification.from_pretrained(model_name3)
        self.jhartmannEmotion_pipeline = pipeline(
            "text-classification",
            model=model3,
            tokenizer=tokenizer3,
            top_k=None
        )

        # (D) Model 3: bhadresh-savani/distilbert-base-uncased-emotion
        model_name4 = "bhadresh-savani/distilbert-base-uncased-emotion"
        tokenizer4 = AutoTokenizer.from_pretrained(model_name4)
        model4 = AutoModelForSequenceClassification.from_pretrained(model_name4)
        self.bhadresh_pipeline = pipeline(
            "text-classification",
            model=model4,
            tokenizer=tokenizer4,
            top_k=None
        )

    def fix_punctuation_spacing(self, text):
        """
        Step 2: Tokenises text using spaCy.
        Merges a numeric token with a following '%' token into one (e.g., "-20" + "%" -> "-20%").
        Lemmatises each token.
        Returns a list of lower-case tokens.
        """
        text = re.sub(r'([,!?;:])([^ \n])', r'\1 \2', text)
        text = re.sub(r'(?<!\d)(\.)(?=[A-Za-z])', r'\1 ', text)
        return text

    def spacy_tokenise_merge(self, text):
        """
        Step 3: Tokenize text using spaCy, merging numeric tokens with '%' into a single token (e.g. '20%').
        Also lemmatize each token. Returns a list of lower-case tokens.
        """
        doc = self.spacy(text)
        merged_tokens = []
        i = 0
        while i < len(doc):
            token = doc[i]
            if (token.like_num or re.match(r'^[-+]?\d+(\.\d+)?$', token.text)) \
               and (i+1<len(doc) and doc[i+1].text=='%'):
                merged_tokens.append((token.text + '%').lower())
                i += 2
            else:
                merged_tokens.append(token.lemma_.lower())
                i += 1
        return merged_tokens

    def get_numeric_votes_from_tokens(self, tokens):
        """
        Step 4: From a list of tokens (from spacy_tokenise_merge), finds tokens representing numeric values.
        Accepts an optional currency symbol and optional multiplier words.
        Triggers a vote only if:
          - The token has a '%' sign, or
          - The preceding token is a directional word or an intent keyword,
            but if the preceding token is already an intent keyword, we skip the numeric vote.
        Returns a dict: {"Buy": count, "Sell": count}.
        """
        pattern = r'^([$£€]?\s*[-+]?\d+(?:[\.,]\d+)?)(%?)$'
        directional_negative = {"down", "fall", "drop", "decline", "decrease", "erase"}
        directional_positive = {"up", "rise", "gain", "increase", "climb", "return"}
        sell_intents = set(self.intent_keywords["Sell"])
        buy_intents = set(self.intent_keywords["Buy"])
        votes = {"Buy": 0, "Sell": 0}
        for i, tok in enumerate(tokens):
            m = re.match(pattern, tok)
            if m:
                try:
                    num_str = m.group(1).strip().replace(',', '.')
                    val = float(num_str)
                except:
                    continue
                # If no explicit sign, check preceding token
                if tok[0] not in "+-":
                    if i>0:
                        prev=tokens[i-1]
                        if prev in buy_intents or prev in sell_intents:
                            continue
                        elif prev in directional_negative:
                            val = -abs(val)
                        elif prev in directional_positive:
                            val = abs(val)
                        else:
                            val = abs(val)
                    else:
                        val=abs(val)
                # Decide if it's triggered as a numeric vote
                trigger=False
                if m.group(2):  # '%'
                    trigger=True
                elif i>0:
                    prev=tokens[i-1]
                    if prev in directional_negative or prev in sell_intents:
                        trigger=True
                    elif prev in directional_positive or prev in buy_intents:
                        trigger=True
                if trigger:
                    if val>0:
                        votes["Buy"]+=1
                    elif val<0:
                        votes["Sell"]+=1
        return votes

    def resolve_ties(self, votes):
        """
        Step 5a: Tie-resolution logic with special cases:
          1. If "Spam" is among the top => return "Spam".
          2. If exactly (Buy,Hold) => ["Buy","Hold"].
          3. If exactly (Sell,Hold) => ["Sell","Hold"].
          4. If exactly (Buy,Hold,Neutral) => ["Buy","Hold"].
          5. If exactly (Sell,Hold,Neutral) => ["Sell","Hold"].
          6. If tie includes (Buy,Sell) => "Neutral".
          7. If tie includes (Buy,Neutral) => "Buy".
          8. If tie includes (Sell,Neutral) => "Sell".
          9. If tie includes (Hold,Neutral) => "Hold".
          10. If 3+ categories tie in any other combination => "Neutral".
          11. If single => that single category.
        """
        max_vote=max(votes.values())
        top_cats=[cat for cat,v in votes.items() if v==max_vote]
        top_set=set(top_cats)
        if "Spam" in top_set:
            return "Spam"
        if top_set=={"Buy","Hold"}:
            return ["Buy","Hold"]
        if top_set=={"Sell","Hold"}:
            return ["Sell","Hold"]
        if top_set=={"Buy","Hold","Neutral"}:
            return ["Buy","Hold"]
        if top_set=={"Sell","Hold","Neutral"}:
            return ["Sell","Hold"]
        if "Buy" in top_set and "Sell" in top_set:
            return "Neutral"
        if "Buy" in top_set and "Neutral" in top_set:
            return "Buy"
        if "Sell" in top_set and "Neutral" in top_set:
            return "Sell"
        if "Hold" in top_set and "Neutral" in top_set:
            return "Hold"
        if len(top_set)>=3:
            return "Neutral"
        if len(top_set)==1:
            return top_set.pop()
        return "Neutral"

    def classify_intent(self, text, k=0.5):
        """
        Step 5b: Classify intent (Buy/Sell/Hold/Neutral/Spam) by combining:
        A) Weighted textual keyword votes (each match = 2 votes)
        B) Numeric votes from tokens (each match = 1 vote)
        Then resolves ties using our priority logic.
        Confidence is computed as:
            (vote ratio) * (1 - exp(-k * total_votes))
        Returns (chosen_intent, confidence)
        """
        tokens = self.spacy_tokenise_merge(text)
        votes = {cat:0 for cat in self.intent_keywords}
        total_votes=0
        TEXTUAL_WEIGHT=2
        NUMERIC_WEIGHT=1

        # Textual
        for cat,keywords in self.intent_keywords.items():
            for kw in keywords:
                count=tokens.count(kw)
                if count>0:
                    votes[cat]+=count*TEXTUAL_WEIGHT
                    total_votes+=count*TEXTUAL_WEIGHT

        # Numeric
        numeric_votes=self.get_numeric_votes_from_tokens(tokens)
        for cat in ["Buy","Sell"]:
            votes[cat]+=numeric_votes.get(cat,0)*NUMERIC_WEIGHT
            total_votes+=numeric_votes.get(cat,0)*NUMERIC_WEIGHT

        # Conflict: if Buy==Sell, add partial to Neutral
        lambda_=0.5
        buy_votes=votes["Buy"]
        sell_votes=votes["Sell"]
        if buy_votes==sell_votes and buy_votes>0:
            conflict_contribution=lambda_*min(buy_votes,sell_votes)
            votes["Neutral"]+=conflict_contribution
            total_votes+=conflict_contribution

        if total_votes==0:
            pseudo_scale=1-math.exp(-k*1)
            return ("Neutral",0.5*pseudo_scale)

        chosen_intent=self.resolve_ties(votes)
        scale=1-math.exp(-k*total_votes)
        if isinstance(chosen_intent,str):
            if chosen_intent in votes:
                ratio=votes[chosen_intent]/total_votes
                confidence=ratio*scale
                return (chosen_intent,confidence)
            else:
                return (chosen_intent,1.0)
        else:
            v1=votes[chosen_intent[0]]
            v2=votes[chosen_intent[1]]
            base_votes=min(v1,v2)
            ratio=base_votes/total_votes
            confidence=ratio*scale
            label_str=" and ".join(chosen_intent)
            return (label_str,confidence)

    def map_emotion_distribution(self, raw_results, mapping):
        """
        Step 6a: Convert raw pipeline output (list of dicts with 'label','score')
        into a dictionary {emotion: probability} based on the provided mapping.
        We sum and re-normalize so that the total is 1.
        """
        dist={e:0.0 for e in self.universal_emotions}
        total=0.0
        for item in raw_results:
            lbl=item['label'].lower()
            score=item['score']
            if lbl in mapping:
                mapped_lbl=mapping[lbl]
                # If it's "love", skip or ignore, because we don't have love in universal set
                if mapped_lbl not in self.universal_emotions:
                    continue
                dist[mapped_lbl]+=score
                total+=score
        epsilon = 1e-6
        for emo in self.universal_emotions:
            dist[emo] = dist.get(emo, 0.0) + epsilon
            total += epsilon
        if total>0:
            for e in dist:
                dist[e]/=total
        return dist

    def aggregate_emotions(self, text, weights=(0.2,0.4,0.4)): #optimised weights
        """
        Step 6b: 
        Aggregates emotion outputs from the three emotion models.
        For each model:
          - Gets raw results via its pipeline.
          - Maps native labels to our universal_emotions using the mapping dict.
          - Re-normalises so each distribution sums to 1.
        Then computes a weighted average distribution such that for each emotion,
        only models that output a nonzero value (above a small epsilon) are used.
        FinalDist(e) = (sum_i [w_i * Dist_i(e)] / (sum_i w_i for models with Dist_i(e) > epsilon)
        Returns a dictionary over universal_emotions.
        """
        # Model 1 distribution from michellejieli
        raw1=self.michellejieli_pipeline(text)[0]
        dist1=self.map_emotion_distribution(raw1,self.mapping_model1)
        # Model 2 distribution from j-hartmann
        raw2=self.jhartmannEmotion_pipeline(text)[0]
        dist2=self.map_emotion_distribution(raw2,self.mapping_model2)
        # Model 3 distribution from bhadresh-savani
        raw3=self.bhadresh_pipeline(text)[0]
        dist3=self.map_emotion_distribution(raw3,self.mapping_model3)

        final_dist={}
        epsilon = 1e-6 
        for e in self.universal_emotions:
            weighted_sum = 0.0
            weight_sum = 0.0
            # For each model, add its contribution only if its output > epsilon.
            if dist1[e] > epsilon:
                weighted_sum += weights[0] * dist1[e]
                weight_sum += weights[0]
            if dist2[e] > epsilon:
                weighted_sum += weights[1] * dist2[e]
                weight_sum += weights[1]
            if dist3[e] > epsilon:
                weighted_sum += weights[2] * dist3[e]
                weight_sum += weights[2]
            if weight_sum > 0:
                final_dist[e] = weighted_sum / weight_sum
            else:
                final_dist[e] = 0.0

        s=sum(final_dist.values())
        if s>0:
            for emo in final_dist:
                final_dist[emo]/=s
        return final_dist

    def dict_to_list(self, dist_dict):
        """
        Helper: Convert dictionary {emotion: prob} to a list [prob_surp, prob_joy, ...]
        matching the order in self.universal_emotions.
        """
        return [dist_dict[e] for e in self.universal_emotions]

    def get_distribution_michellejieli_list(self, text):
        """
        Step 6c: 
        Return Model1 distribution as a LIST for computing cosine similarity.
        """
        raw=self.michellejieli_pipeline(text)[0]
        dist_dict=self.map_emotion_distribution(raw,self.mapping_model1)
        return self.dict_to_list(dist_dict)

    def get_distribution_jhartmann_list(self, text):
        """
        Step 6c:
        Return Model2 distribution as a LIST for computing cosine similarity.
        """
        raw=self.jhartmannEmotion_pipeline(text)[0]
        dist_dict=self.map_emotion_distribution(raw,self.mapping_model2)
        return self.dict_to_list(dist_dict)

    def get_distribution_bhadresh_list(self, text):
        """
        Step 6c:
        Return Model 3 distribution as a LIST for computing cosine similarity.
        """
        raw = self.bhadresh_pipeline(text)[0]
        dist_dict = self.map_emotion_distribution(raw, self.mapping_model3)
        return self.dict_to_list(dist_dict)

    def cosine_similarity(self, vec1, vec2):
        """
        Step 7a:
        Standard cosine similarity between two lists of floats.
        """
        dot=sum(a*b for a,b in zip(vec1,vec2))
        norm1=math.sqrt(sum(a*a for a in vec1))
        norm2=math.sqrt(sum(b*b for b in vec2))
        if norm1==0 or norm2==0:
            return 0.0
        return dot/(norm1*norm2)

    def compute_emotion_confidence(self, text):
        """
        Step 7b:
        Compute overall emotion confidence by averaging pairwise cosine similarity 
        among the 3 model distributions (converted to lists).
        """
        d1=self.get_distribution_michellejieli_list(text)
        d2=self.get_distribution_jhartmann_list(text)
        d3=self.get_distribution_bhadresh_list(text)
        sim12=self.cosine_similarity(d1,d2)
        sim13=self.cosine_similarity(d1,d3)
        sim23=self.cosine_similarity(d2,d3)
        return (sim12+sim13+sim23)/3.0

    def nlp(self, text):
        """
        Step 7c: 
        Master analysis method that:
          - Fixes punctuation,
          - Runs FinBERT for financial sentiment,
          - Aggregates emotion distribution and computes overall emotion confidence
          - Classifies intent with confidence
          - Prints all results
        """
        text = self.fix_punctuation_spacing(text)
        #print(f"\n\033[1mTweet:\033[0m {text}")

        # 1. FinBERT
        finbert_results=self.finbert_pipeline(text)
        finbert_scores = {res['label']: res['score'] for res in finbert_results[0]}
        positive = finbert_scores.get("positive", 0.0)
        negative = finbert_scores.get("negative", 0.0)
        neutral = finbert_scores.get("neutral", 0.0)
        #print("\n\033[1mFinBERT Sentiment:\033[0m")
        #for result in finbert_results[0]:
            #print(f"  {result['label']}: {result['score']:.4f}")

        # 2a. Aggregated Emotions
        w1, w2, w3 = (0.2, 0.4, 0.4)
        agg_emotions=self.aggregate_emotions(text, weights=(w1, w2, w3))
        surprise = agg_emotions.get("surprise", 0.0)
        joy = agg_emotions.get("joy", 0.0)
        anger = agg_emotions.get("anger", 0.0)
        fear = agg_emotions.get("fear", 0.0)
        sadness = agg_emotions.get("sadness", 0.0)
        disgust = agg_emotions.get("disgust", 0.0)
        emotion_conf=self.compute_emotion_confidence(text)
        #print("\n\033[1mAggregated Emotion Distribution:\033[0m")
        #for emo in self.universal_emotions:
            #print(f"  {emo}: {agg_emotions[emo]:.4f}")
        #emotion_conf=self.compute_emotion_confidence(text)
        #print(f"(confidence: {emotion_conf:.4f})")

        # 3. Intent
        intent_label,intent_conf=self.classify_intent(text, k=self.k_intent)
        #print(f"\n\033[1mIntent-based Sentiment:\033[0m {intent_label}")
        #print(f"(confidence: {intent_conf:.4f})")

        return [positive, negative, neutral, surprise, joy, anger, fear, sadness, disgust, emotion_conf, intent_label, intent_conf]


###############################################################################
# Example usage
###############################################################################
#if __name__=="__main__":
    #analyzer=SentimentEmotionAnalyzer()
    #example_text="$NVDA, the most valuable company in the world finished down 8.7%. It lost $250 billion in market cap. Unusual."
    #analyzer.nlp(example_text)


#read excel
"""
import pandas as pd
from datetime import datetime
data_path = "C:/Users/zly/Desktop/data.xlsx"
df = pd.read_excel(data_path)
df["Date"] = df["Created_At"].apply(lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y").strftime("%Y/%m/%d"))

analyzer = SentimentEmotionAnalyzer()

columns = [
    "Positive", "Negative", "Neutral",
    "Surprise", "Joy", "Anger", "Fear", "Sadness", "Disgust",
    "Emotion Confidence", "Intent Sentiment", "Confidence"
]

results = [analyzer.nlp(text) for text in df["Text"].astype(str)]

df[columns] = results
df = df[df["Emotion Confidence"] >= 0.5]

aggregated_df = df.groupby(["Company", "Date"]).agg({
    "Positive": "mean", "Negative": "mean", "Neutral": "mean",
    "Surprise": "mean", "Joy": "mean", "Anger": "mean", "Fear": "mean",
    "Sadness": "mean", "Disgust": "mean",
    "Intent Sentiment": lambda x: x.value_counts().idxmax() 
}).reset_index()

output_path = "C:/Users/zly/Desktop/output1.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Raw Data", index=False)
    aggregated_df.to_excel(writer, sheet_name="Aggregated Data", index=False)

print(f"Done. Results are saved to  {output_path}")"
"""

#read sql
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

analyzer = SentimentEmotionAnalyzer()

db_user = "postgres"
db_password = "qf5214"
db_host = "134.122.167.14"
db_port = 5555
db_name = "QF5214"

engine = create_engine(
    f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?options=-csearch_path%3Ddatacollection"
)

#读取前 250 行数据
query = """
SELECT Company, Text, Created_at 
FROM x_batch 
ORDER BY created_at ASC 
LIMIT 250;
"""
df = pd.read_sql(query, engine)

df["Date"] = df["created_at"].apply(lambda x: x.strftime("%Y/%m/%d"))

columns = [
    "Positive", "Negative", "Neutral",
    "Surprise", "Joy", "Anger", "Fear", "Sadness", "Disgust",
    "Emotion Confidence", "Intent Sentiment", "Confidence"
]
results = [analyzer.nlp(text) for text in df["text"].astype(str)]
df[columns] = results
df = df[df["Emotion Confidence"] >= 0.5]

aggregated_df = df.groupby(["company", "Date"]).agg({
    "Positive": "mean", "Negative": "mean", "Neutral": "mean",
    "Surprise": "mean", "Joy": "mean", "Anger": "mean", "Fear": "mean",
    "Sadness": "mean", "Disgust": "mean",
    "Intent Sentiment": lambda x: x.value_counts().idxmax()
}).reset_index()

output_path = "C:/Users/zly/Desktop/output2.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Sentiment Raw Data", index=False)
    aggregated_df.to_excel(writer, sheet_name="Sentiment Aggregated Data", index=False)

print(f"saved {output_path}")

# 写入 PostgreSQL 两张表
# df.to_sql("sentiment_raw_data", engine, schema="nlp", if_exists="replace", index=False)
# aggregated_df.to_sql("sentiment_aggregated_data", engine, schema="nlp", if_exists="replace", index=False)
# print("saved: sentiment_raw_data and sentiment_aggregated_data")

#自动更新
"""
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import schedule
import time

analyzer = SentimentEmotionAnalyzer()

engine = create_engine(
    "postgresql+psycopg2://postgres:qf5214@134.122.167.14:5555/QF5214?options=-csearch_path%3Ddatacollection"
)

#每天执行一次的函数
def run_daily_sentiment_analysis():
    print("start analyzing")


    query = """
    SELECT Company, Text, Created_at
    FROM x_batch
    WHERE created_at >= CURRENT_DATE AND created_at < CURRENT_DATE + INTERVAL '1 day'
    ORDER BY created_at ASC
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        print("No data today, skip")
        return

    df["Date"] = df["created_at"].apply(lambda x: x.strftime("%Y/%m/%d"))

    columns = [
        "Positive", "Negative", "Neutral",
        "Surprise", "Joy", "Anger", "Fear", "Sadness", "Disgust",
        "Emotion Confidence", "Intent Sentiment", "Confidence"
    ]
    results = [analyzer.nlp(text) for text in df["text"].astype(str)]
    df[columns] = results
    df = df[df["Emotion Confidence"] >= 0.5]

    aggregated_df = df.groupby(["company", "Date"]).agg({
        "Positive": "mean", "Negative": "mean", "Neutral": "mean",
        "Surprise": "mean", "Joy": "mean", "Anger": "mean", "Fear": "mean",
        "Sadness": "mean", "Disgust": "mean",
        "Intent Sentiment": lambda x: x.value_counts().idxmax()
    }).reset_index()

    df.to_sql("sentiment_raw_data", engine, schema="nlp", if_exists="append", index=False)
    aggregated_df.to_sql("sentiment_aggregated_data", engine, schema="nlp", if_exists="append", index=False)

    print("completed analyzing")

# 每天 6:00 自动执行一次
schedule.every().day.at("06:00").do(run_daily_sentiment_analysis)

while True:
    schedule.run_pending()
    time.sleep(60)

"""
