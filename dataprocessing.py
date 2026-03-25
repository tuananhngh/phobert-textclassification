import json
import re
import pandas as pd


def read_data(path_to_file):
    with open(path_to_file, encoding="utf-8-sig") as f:
        text = json.loads(f.read())

    seq = [entry["subject"] for entry in text["data_direction"]]
    lab = [entry["category"] for entry in text["data_direction"]]

    return pd.DataFrame({"Sequence": seq, "Label": lab})


def regex_sentence(s):
    s = re.sub(r"((www\.[^\s]+)|(https://[^\s]+))", "URL", s)
    s = re.sub(r"V/v", "", s)
    s = re.sub(r"v/v", "", s)
    s = re.sub(r"Về việc", "", s)
    s = re.sub(r'[-–()/"#@;:<>{}`+=~|.!?,&""%*⋅…]', " ", s)
    s = re.sub(r"\b\d+\b", "", s)
    s = re.sub(r"\b[BCEXHVICJFQPKcvhđmgbs]\b", "", s)
    s = s.replace("\n", "").replace("\r", "").replace("\\", "")
    s = s.strip()
    s = " ".join(s.split())
    s = s.lower()
    return s
