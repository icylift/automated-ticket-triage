"""
Simple Flask service that accepts JSON ticekts and returns triage suggestions:
-category (rule or ML)
-confidence
-priority
-assignee (simple load-based selection)
"""

import os 
import joblib
from flask import Flask, request, jsonify
from datetime import datetime
from rules import apply_rules
import re

# Load model (trained previously)
MODEL_PATH = os.path.join("models", "triage_pipline.joblib")
if not os.path.exists(MODEL_PATH):
  raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run models/train_model.py first")

pipline = joblib.load(MODEL_PATH)

app = Flask(__name__)

# Simple in-memory agent pool for the demo. Replace with DB or servies in prod
AGENTS = [
  {"id": "Alice", "skills": ["Password", "Hardware"], "load": 2},
  {"id": "Bob", "skills": ["Network", "Incident"], "load": 1},
  {"id": "Cara", "skills": ["ServieRequest", "Hardware"], "load": 0},
]

# Category --> default priority mapping, we'll boost if key words appear
CATEGORY_PRIORITY = {
  "Password": "High",
  "Hardware": "Low",
  "Network": "High",
  "ServiceRequest": "Low",
  "Incident": "Critical"
}

PRIORITY_KEYWORDS = {
  "Critical": re.compile(r"(production|data loss|cannot work|can't work|outage|ransom|encrypted|critical)", re.I),
  "High": re.compile(r"(urgent|asap|immediately|priority)", re.I)
}


