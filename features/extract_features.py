from features import lexical
from utils.utils import *


def init_dict():
    feature_dict = dict()
    for f in lexical.feature_name:
        feature_dict[f] = dict()
        for seg in lexical.seg_feature.keys():
            feature_dict[f][seg] = dict()
    return feature_dict


def extract_all_features(doc, doc_name):
    segmentations = ['g20-10', 'g30-10', 's']
    feature_dict = init_dict()
    for seg in segmentations:
        segments = segment(doc, seg)
        lexical_features = lexical.extract_features(seg, segments)
        for i, f_indx in enumerate(lexical.seg_feature[seg]):
            feature_dict[lexical.feature_name[f_indx]][seg][doc_name] = {'Feature': lexical_features[:, i]}
    return feature_dict


if __name__ == '__main__':
    doc = preprocess(read_file('../docs/1.txt'))
    print(extract_all_features(doc, '1'))
    # get_predict_data('g09-00', extract_all_features(doc, '1'), '1')
