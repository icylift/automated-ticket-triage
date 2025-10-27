# models/train_model.py
"""
train a simple TF-IDF + logisticRegression pipeline and save to disk
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Config

DATA_PATH = os.path.join("..", "data", "sample_tickets.csv")
MODEL_OUT = os.path.join("..", "models", "triage_pipline.joblib")

def load_data(path):
  df = pd.read_csv(path)
  # combine subject + body --> text used for classification
  df['text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
  return df

def train_and_save(df):
  x = df['text']
  y = df['category']

  #split for quick evaluation
  X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42, stratify=y )

# Pipline: TF-IDF  vectorizer + Logistic regression classifier 
  pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
  ]) 

  pipeline.fit(X_train, y_train)

  # quick evaluation 
  preds = pipeline.predict(X_test)
  print("=== Classification report on test set ===")
  print(classification_report(y_test, preds))

  # ensure models directory exist then saves
  os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
  joblib.dump(pipeline, MODEL_OUT)
  print(f"Saved pipline to {MODEL_OUT}")

if __name__ == "__main__":
  df = load_data(DATA_PATH)
  train_and_save(df)
