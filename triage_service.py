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

def pick_asignee(category):
  """Select an agent that has the category skill and lowest load. Fallback to the lowest load overall"""
  candidates = [a for a in AGENTS if category in a['skills']]
  if not candidates:
    candidates = AGENTS
  chosen = min(candidates, key=lambda a: a['load'])
  # increment load to stimulate assignment (persist in production)
  chosen['load'] += 1
  return chosen['id']

def compute_priority(category, subject, body):
  base = CATEGORY_PRIORITY.get(category, "Medium")
  text = (subject or "") + " " + (body or "")
  if PRIORITY_KEYWORDS["Critical"].search(text):
    return "Critical"
  if PRIORITY_KEYWORDS["High"].search(text):
    return "High"
  return  base

@app.route("/triage", methods=["POST"])
def triage_ticket():
  payload = request.get_json(force=True)
  subject = payload.get("subject", "")
  body = payload.get("body", "")
  ticket_id = payload.get("ticket_id", f"local-{int(datetime.utcnow().timestamp())}")

  # 1) Try rules first (High-precision)
  rule_category = apply_rules(subject, body)
  if rule_category:
    category = rule_category
    method = "rule"
    confidence = 1.0
  else:
    #2 ) ML fallback
    text = (subject or "") + " " + (body or "")
    pred = pipline.predict([text])[0]
    proba = pipline.predict_proba([text])[0]
    labels = list(pipline.classes_)
    confidence = float(proba[labels.index(pred)])
    category = pred
    method = "ml"

  #3 ) compute priority
  priority = compute_priority(category, subject, body)

  #4 ) pick assignee
  assignee = pick_asignee(category)

  # prepare response 
  resp = {
    "ticket_id": ticket_id,
    "category": category,
    "method": method,
    "confidence": round(confidence, 3),
    "priority": priority,
    "assignee": assignee,
    "triage_at": datetime.utcnow().isoformat() + "Z"
  }

  # in prod: save triage decision to DB + call ticketing API to update tickets.
  return jsonify(resp), 200

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
  



