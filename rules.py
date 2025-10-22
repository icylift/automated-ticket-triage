import re

# patterns for exact or near exact matches that we want to catch with 100% precision
RULES = [
  ("Password", re.compile(r"(forgot.*pass|reset password|password reset|can't login|cannot login|forgot pasword|reset my password)", re.I)),
  ("Hardware", re.compile(r"(printer|paper jam|wont print|batery|ssd|hdd|fan noise|overheating|battery swelling)", re.I)),
  ("Network", re.compile(r"(vpn|no internet|internet down|cannot reach|network down|latnecy|disconnect|outage|email bounce|smtp 550)", re.I)),
  ("ServiceRequest", re.compile(r"(request to install|please install|request install|access to|permission to access|need access to)", re.I)),
  ("Incident", re.compile(r"(ransomware|encrypted files|malware|viruse|bsod|blue screen|data leak|data exfiltration)", re.I)),
]

def apply_rules(subject: str, body: str):
  """Return category string if a rule matches; otherwise None. """
  text = (subject or "") + " " + (body or "")
  for category, pattern in RULES:
    if pattern.search(text):
      return category
  return None
