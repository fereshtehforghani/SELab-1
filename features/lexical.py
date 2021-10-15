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