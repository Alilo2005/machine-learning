import nltk 
from nltk.tokenize import word_tokenize as wt
from nltk.util import ngrams as ng

sample_text = 'I am learning NLP(Natural Language Processing)'
tokens = wt(sample_text)

unigrams = list(ng(tokens, 1))
bigrams = list(ng(tokens, 2))
trigrams = list(ng(tokens, 3))

print("Unigrams: ", unigrams)
print("Bigrams: ", bigrams)
print("Trigrams: ", trigrams)