from textblob import TextBlob
text = 'I love progamming and machine learnig.'
blob = TextBlob(text)
corrected_text = blob.correct()
sentiment = blob.sentiment
print("Original text: ", text)
print("Corrected text: ", corrected_text)
print("Sentiment: ", sentiment)