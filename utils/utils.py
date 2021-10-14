import re
import spacy
import textacy
import numpy as np
en = textacy.load_spacy_lang('en_core_web_sm')
en.max_length = 10000000

segment_patter = re.compile('([sg])(([0-9]{2})-([0-9]{2}))?')


def segment(doc, segmentation):
    token, _, token_count, overlap = segment_patter.search(segmentation).groups()
    if token == 's':
        return list(doc.sents)
    if token == 'g':
        token_count, overlap = int(token_count), int(overlap)
        length = len(doc)
        segments, start, end = [], 0, 0
        while end != length:
            end = min(length, start + token_count)
            segments.append(doc[start:end])
            start += token_count - overlap
        return segments
    raise ValueError('Token is either s for sentences or g for n-grams!')



