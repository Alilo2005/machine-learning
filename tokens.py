import nltk
nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize

sample_text = "Now i become death, the destroyer of worlds"
tokens = word_tokenize(sample_text.lower())
print("Tokens: ", tokens)