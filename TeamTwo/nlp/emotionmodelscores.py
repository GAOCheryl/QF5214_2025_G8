import pandas as pd
import re
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report

# Define the universal emotion set for evaluation.
# Our universal set: sadness, joy, anger, fear, disgust, surprise
universal_emotions = ["sadness", "joy", "anger", "fear", "disgust", "surprise"]

# --- Mapping dictionaries for each model ---
# For Model 1: michellejieli/emotion_text_classifier
mapping_model1 = {
    "surprise": "surprise",
    "joy": "joy",
    "anger": "anger",
    "fear": "fear",
    "sadness": "sadness",
    "disgust": "disgust"
}

# For Model 2: cardiffnlp/twitter-roberta-base-emotion-multilabel-latest
mapping_model2 = {
    "joy": "joy",
    "optimism": "joy",
    "surprise": "surprise",
    "fear": "fear",
    "anger": "anger",
    "pessimism": "sadness",
    "disgust": "disgust",
    "sadness": "sadness"
}

# For Model 3: j-hartmann/emotion-english-distilroberta-base
mapping_model3 = {
    "surprise": "surprise",
    "joy": "joy",
    "anger": "anger",
    "fear": "fear",
    "sadness": "sadness",
    "disgust": "disgust"
}

# For Model 4: bhadresh-savani/distilbert-base-uncased-emotion
mapping_model4 = {
    "surprise": "surprise",
    "joy": "joy",
    "anger": "anger",
    "fear": "fear",
    "sadness": "sadness",
    "love": "love"  # This model outputs 'love'
}

# Dataset mapping: convert numeric labels (0-5) to emotion strings.
# Both datasets use: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)
dataset_mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def load_pipeline(model_name, multi_label=False):
    """
    Loads and returns a Hugging Face pipeline for text classification given a model name.
    If multi_label is True, the model is assumed to be multi-label.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

def map_emotion_distribution(raw_results, mapping):
    """
    Maps the raw output (list of dicts with 'label' and 'score')
    from a model's pipeline to the universal_emotions using the provided mapping.
    If the output is nested, it is flattened.
    The resulting probabilities are re-normalized to sum to 1.
    Returns a dictionary {emotion: probability, ...}.
    """
    if raw_results and isinstance(raw_results[0], list):
        raw_results = raw_results[0]
    dist = {e: 0.0 for e in universal_emotions}
    total = 0.0
    for item in raw_results:
        lbl = item['label'].lower()
        score = item['score']
        if lbl in mapping:
            mapped_lbl = mapping[lbl]
            if mapped_lbl in universal_emotions:
                dist[mapped_lbl] += score
                total += score
    if total > 0:
        for e in dist:
            dist[e] /= total
    return dist

def evaluate_model_on_dataset_batch(model_name, mapping, multi_label, csv_path, batch_size=64):
    """
    Evaluates a single emotion model on a CSV file (with 'text' and 'label' columns) using batch processing.
    Rows with true label "love" are skipped (since our universal set does not include "love").
    Returns lists of true labels and predicted labels.
    """
    df = pd.read_csv(csv_path)
    true_labels = []
    predicted_labels = []
    pipe = load_pipeline(model_name, multi_label=multi_label)
    
    texts = df['text'].tolist()
    labels_numeric = df['label'].tolist()
    
    # Process in batches.
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels_numeric[i:i+batch_size]
        raw_outputs = pipe(batch_texts)
        for j, raw in enumerate(raw_outputs):
            true_label = dataset_mapping.get(batch_labels[j], "").lower()
            if true_label == "love":  # skip rows with "love"
                continue
            dist = map_emotion_distribution(raw, mapping)
            if dist:
                pred_label = max(dist, key=dist.get)
            else:
                pred_label = "unknown"
            true_labels.append(true_label)
            predicted_labels.append(pred_label)
    
    return true_labels, predicted_labels

def evaluate_model(model_name, mapping, multi_label, csv_path, batch_size=64):
    """
    Evaluates a single model on a CSV file and prints the classification report.
    """
    true_labels, predicted_labels = evaluate_model_on_dataset_batch(model_name, mapping, multi_label, csv_path, batch_size=batch_size)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(true_labels, predicted_labels))
    print("=" * 60)

if __name__ == "__main__":
    # Use a single CSV file for evaluation; adjust the file path as needed.
    csv_file = r"C:\Users\jiazh\Downloads\test_data.csv"
    
    print("Evaluating Emotion Models (skipping rows with 'love') using batch processing:\n")
    evaluate_model("michellejieli/emotion_text_classifier", mapping_model1, multi_label=False, csv_path=csv_file, batch_size=64)
    evaluate_model("cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", mapping_model2, multi_label=True, csv_path=csv_file, batch_size=64)
    evaluate_model("j-hartmann/emotion-english-distilroberta-base", mapping_model3, multi_label=False, csv_path=csv_file, batch_size=64)
    evaluate_model("bhadresh-savani/distilbert-base-uncased-emotion", mapping_model4, multi_label=False, csv_path=csv_file, batch_size=64)
