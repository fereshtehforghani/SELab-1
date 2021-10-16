import spacy
import textacy
import numpy as np
from utils.utils import normalize
from collections import Counter
from scipy.stats import entropy
from nltk.stem.snowball import SnowballStemmer
from functools import lru_cache
# from cytoolz import itertoolz

stemmer = SnowballStemmer(language='english')


def to_terms_list(doc, ngrams=1, **kwargs):
    normalize = kwargs.get('normalize', 'lemma')
    del kwargs['normalize']
    terms = textacy.extract.ngrams(doc, ngrams, **kwargs)
    if normalize == 'lemma':
        for term in terms:
            yield term.lemma_
    elif normalize == 'lower':
        for term in terms:
            yield term.lower_
    elif callable(normalize):
        for term in terms:
            yield normalize(term)
    else:
        for term in terms:
            yield term.text

@lru_cache(maxsize=32)
def bag_of_terms(doc, ngrams=1, **kwargs):
    kwargs['normalize'] = stem
    if isinstance(doc, spacy.tokens.doc.Doc):
        return doc._.to_bag_of_terms(ngrams=ngrams, as_strings=True, **kwargs)
    return Counter(to_terms_list(doc, ngrams, **kwargs))


def stem(token):
    return stemmer.stem(token.lemma_)

def brunet_measure(bot, n, v):
    a = 0.17
    if n == 0 or n == 1:
        return 0
    return (v - a) / (np.log(n))


def hapax_dislegemena(bot, n, v):
    v2 = len([word for word, count in bot.items() if count == 2])
    if n == 0:
        return 0
    return v2 / n