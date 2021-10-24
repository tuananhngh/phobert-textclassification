import json
import pandas as pd
import os
import re
from collections import defaultdict


def read_data(path_to_file):
    with open(path_to_file, encoding='utf-8-sig') as f:
        text = json.loads(f.read())

    seq = [text["data_direction"][i]["subject"] for i in range(len(text["data_direction"]))]
    lab = [text["data_direction"][i]["category"] for i in range(len(text["data_direction"]))]

    data_pd = pd.DataFrame({"Sequence": seq, "Label": lab})
    return data_pd


def find_accronym(data_file):
    accronym = []
    for i in range(data_file.shape[0]):
        sent = data_file.iloc[i]["Sequence"]
        sent_acc = re.findall(r'[A-ZĐ]{2,}', sent)
        if sent_acc:
            for ele in sent_acc:
                accronym.append(ele)
    return set(accronym)


def regex_sentence(s):
    s = re.sub('((www\.[^s]+)|(https://[^\s]+))', 'URL', s)  # replace url
    s = re.sub("V/v", "", s)
    s = re.sub("v/v", "", s)
    s = re.sub("Về việc", "", s)
    s = re.sub(r'[-–()/"#@;:<>{}`+=~|.!?,&“”%*⋅…]', ' ', s)
    s = re.sub(r"\b\d+\b", '', s)  # remove number, date, etc...
    #s = re.sub("TTg", "", s)
    #s = re.sub("CTr]", "", s)
    #s = re.sub(r'\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠ]\b', "", s)  #remove single uppercase character
    s = re.sub(r'\b[BCEXHVICJFQPKcvhđmgbs]\b', "", s)
    #s = re.sub(r'[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠ]{2,}', "", s)  #remove 2 consecutive uppercase character
    #s = re.sub('[\n]+', '', s)  #remove white space
    s = s.replace('\n', '').replace('\r', '').replace("\\", "")
    s = s.strip()
    s = ' '.join(word for word in s.split())  #
    s = s.lower()
    return s


def get_single_letter(s):
    return [word for word in s.split() if len(word) == 1]


def single_letters(data):
    single = defaultdict(list)
    for i,seq in enumerate(data["Sequence"]):
        ok = get_single_letter(seq)
        for lt in ok:
            single[lt].append(i)
    return single







