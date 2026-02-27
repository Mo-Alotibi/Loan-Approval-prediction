from nltk.stem import PorterStemmer
import nltk

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

stemmer = PorterStemmer()

def apply_stemming(tokens: list) -> list:
    """Applies PorterStemmer to a list of tokens."""
    return [stemmer.stem(token) for token in tokens]
