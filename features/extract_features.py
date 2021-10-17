from features import lexical
from utils.utils import *

def init_dict():
    feature_dict = dict()
    for f in lexical.feature_name:
        feature_dict[f] = dict()
        for seg in lexical.seg_feature.keys():
            feature_dict[f][seg] = dict()
    return feature_dict
