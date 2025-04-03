#BATCH FOR NLP

import re
import math
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Tuple


class SentimentEmotionAnalyzerBatch(SentimentEmotionAnalyzer):

    def batch_nlp(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        
        Change the nlp process into batch process:
            texts: List of input texts to analyze
            batch_size: Number of texts to process simultaneously
            
        """
        # Initialize results list
        results = []
        
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
        return results

    def _process_batch(self, batch: List[str]) -> List[Dict]:
        batch_results = []
        
        preprocessed_texts = [self.fix_punctuation_spacing(text) for text in batch]
        
        # Process FinBERT in batch
        with torch.no_grad():
            finbert_batch = self.finbert_pipeline(preprocessed_texts)
        
        # Process emotions in batch (each model separately)
        with torch.no_grad():
            michellejieli_batch = self.michellejieli_pipeline(preprocessed_texts)
            jhartmann_batch = self.jhartmannEmotion_pipeline(preprocessed_texts)
            bhadresh_batch = self.bhadresh_pipeline(preprocessed_texts)
        
        # Process each text in the batch
        for i, text in enumerate(preprocessed_texts):
            # 1. FinBERT results
            finbert_scores = {res['label'].lower(): res['score'] for res in finbert_batch[i]}
            
            # 2. Aggregated Emotions
            # Map each model's output to universal emotions
            # Model 1 distribution from michellejieli
            dist1 = self.map_emotion_distribution(michellejieli_batch[i], self.mapping_model1)
            # Model 2 distribution from j-hartmann
            dist2 = self.map_emotion_distribution(jhartmann_batch[i], self.mapping_model2)
            # Model 3 distribution from bhadresh-savani
            dist3 = self.map_emotion_distribution(bhadresh_batch[i], self.mapping_model3)
            
            # Weighted average
            weights = (0.2, 0.4, 0.4)
            epsilon = 1e-6
            agg_emotions = {}
            for e in self.universal_emotions:
                weighted_sum = 0.0
                weight_sum = 0.0
                if dist1[e] > epsilon:
                    weighted_sum += weights[0] * dist1[e]
                    weight_sum += weights[0]
                if dist2[e] > epsilon:
                    weighted_sum += weights[1] * dist2[e]
                    weight_sum += weights[1]
                if dist3[e] > epsilon:
                    weighted_sum += weights[2] * dist3[e]
                    weight_sum += weights[2]
                agg_emotions[e] = weighted_sum / weight_sum if weight_sum > 0 else 0.0
            
            # Normalize
            s = sum(agg_emotions.values())
            if s > 0:
                for emo in agg_emotions:
                    agg_emotions[emo] /= s
            
            # 3. Emotion confidence
            d1 = self.dict_to_list(dist1)
            d2 = self.dict_to_list(dist2)
            d3 = self.dict_to_list(dist3)
            sim12 = self.cosine_similarity(d1, d2)
            sim13 = self.cosine_similarity(d1, d3)
            sim23 = self.cosine_similarity(d2, d3)
            emotion_conf = (sim12 + sim13 + sim23) / 3.0
            
            # 4. Intent classification
            intent_label, intent_conf = self.hybrid_classify_intent(text, k=self.k_intent)
            
            # Store results
            result = {
                'text': text,
                'finbert_scores': finbert_scores,
                'agg_emotions': agg_emotions,
                'emotion_confidence': emotion_conf,
                'intent_label': intent_label,
                'intent_confidence': intent_conf,
                'numeric_features': [
                    finbert_scores.get("positive", 0.0), 
                    finbert_scores.get("negative", 0.0), 
                    finbert_scores.get("neutral", 0.0),
                    agg_emotions["surprise"], 
                    agg_emotions["joy"], 
                    agg_emotions["anger"], 
                    agg_emotions["fear"],
                    agg_emotions["sadness"], 
                    agg_emotions["disgust"],
                    emotion_conf, 
                    intent_label, 
                    intent_conf
                ]
            }
            batch_results.append(result)
            
        return batch_results

###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    analyzer = SentimentEmotionAnalyzerBatch()
    example_texts=df.read_csv('filtered_sample_data.csv')
    results = analyzer.batch_nlp(example_texts)