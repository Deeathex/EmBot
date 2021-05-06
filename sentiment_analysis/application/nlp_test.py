from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("going", pos='v'))
print(lemmatizer.lemmatize("obviously", pos='a'))
print(lemmatizer.lemmatize("obviously", pos='r'))  # adverb
