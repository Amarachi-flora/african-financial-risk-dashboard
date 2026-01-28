# test_vader.py
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("✅ VADER successfully imported!")
    
    analyzer = SentimentIntensityAnalyzer()
    test_text = "This service is excellent and very helpful!"
    scores = analyzer.polarity_scores(test_text)
    print(f"✅ Test analysis: {scores}")
    print(f"✅ Compound score: {scores['compound']}")
except Exception as e:
    print(f"❌ Error: {e}")
    
    
    import nltk
nltk.download('vader_lexicon')  # Run this line once
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(str(text))
    return scores['compound']  # Returns a compound score from -1 to 1