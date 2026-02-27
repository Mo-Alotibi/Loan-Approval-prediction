import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize(text: str) -> list:
    text = clean_text(text)
    return text.split()

def build_vocab(tokenized_texts: list, max_vocab_size=5000) -> dict:
    freq = {}
    for tokens in tokenized_texts:
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
    sorted_vocab = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]
    return {word: idx + 2 for idx, (word, _) in enumerate(sorted_vocab)}  # 0:PAD, 1:UNK