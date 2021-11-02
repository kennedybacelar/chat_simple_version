import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import numpy as np

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')


def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())

def bag_of_words(tokenized_sencente, words):
  sentence_words = [stem(word) for word in tokenized_sencente]

  bag = np.zeros(len(words), dtype=np.float32)
  for idx, w in enumerate(words):
    if w in sentence_words:
      bag[idx] = 1
  
  return bag