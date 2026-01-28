import sys
print("VS Code Python:", sys.executable)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("✅ SUCCESS: VADER is now found!")
except Exception as e:
    print("❌ ERROR:", e)