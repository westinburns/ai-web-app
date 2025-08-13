from textblob import TextBlob

test_phrases = [
    "I love this AI assistant, it's amazing!",
    "This is not what I wanted at all.",
    "The weather is 72 degrees.",
    "Your last response was okay, but a bit slow."
]

print("--- Running Sentiment Analysis Demo ---")

for phrase in test_phrases:
    # Create a TextBlob object
    blob = TextBlob(phrase)
    
    # Get the sentiment object
    sentiment = blob.sentiment
    
    print(f"\nPhrase: '{phrase}'")
    print(f"  -> Polarity: {sentiment.polarity:.2f}  (-1=Negative, 1=Positive)")
    print(f"  -> Subjectivity: {sentiment.subjectivity:.2f}  (0=Objective, 1=Subjective)")