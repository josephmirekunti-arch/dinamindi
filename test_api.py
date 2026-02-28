import urllib.request
import json

url = "http://127.0.0.1:8004/api/predictions"
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        
    predictions = data.get('data', [])
    print(f"Total Predictions Available: {len(predictions)}")
    
    for i, p in enumerate(predictions[:30]):
        print(f"{i+1}. {p['home_team']} vs {p['away_team']}")
        
except Exception as e:
    print(f"Error: {e}")
