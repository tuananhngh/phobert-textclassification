import pandas as pd
from collections import defaultdict


def get_unique_label(dataset):
    dict_text = defaultdict(list)
    for k, v in zip(dataset["Sequence"], dataset["Label"]):
        if v not in dict_text[k]:
            #print(f"{v} not in {dict_text[k]}")
            dict_text[k].append(v)
        else:
            #print(f"{v} in {dict_text[k]}")
            pass
    return dict_text


def segmenter(sent, segm):
    sentence = segm.tokenize(sent)[0]
    return " ".join(sentence)


def data_label_dict(dataframe, segm):
    unseg_dict = get_unique_label(dataframe)
    seg_dict = dict()  #defaultdict(list)
    for k, v in unseg_dict.items():
        seg_dict[segmenter(k, segm)] = v
    return seg_dict


def create_pd_dummies_label(dictionary, label_list):
    empty_pd = pd.DataFrame(index=range(len(dictionary)), columns=["Sequence", *label_list])
    for i, k in enumerate(dictionary.keys()):
        empty_pd.iloc[i]["Sequence"] = k
        for lab in dictionary[k]:
            empty_pd.iloc[i][lab] = 1
    empty_pd = empty_pd.fillna(0)
    return empty_pd
