import numpy as np
from utils.utils import normalize


def mean_word_length(seg):
    words = [token for token in seg if not token.is_punct]
    if not words:
        return 0
    length = sum([len(word) for word in words])
    return length / len(words)


def punctuation_count(seg):
    puncts = [token for token in seg if token.is_punct]
    return len(puncts) / len(seg)
