# models/train_model.py
"""
train a simple TF-IDF + logisticRegression pipeline and save to disk
"""

import os
import pandas as pd
from sklearn.feature_extraction.txt import TfidfVectorizer