import numpy as np
import torch
import random
from vncorenlp import VnCoreNLP
from dataprocessing import *
from segmenting import *
from model import *
from trainer import *
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

BATCH_SIZE = 32
EPOCHS = 5


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)


def bert_preprocessing(data):
    inputs_id = []
    att_masks = []
    tk = tokenizer()
    for sent in data:
        encoded_sent = tk.encode_plus(text=sent, add_special_tokens=True,
                                      return_attention_mask=True,
                                      max_length=114,
                                      padding='max_length', )
        #print(len(encoded_sent.get('input_ids')))
        inputs_id.append(encoded_sent.get('input_ids'))
        att_masks.append(encoded_sent.get('attention_mask'))
    inputs_id = torch.tensor(inputs_id)
    att_masks = torch.tensor(att_masks)
    return inputs_id, att_masks


if __name__ == '__main__':
    #read data from source
    path_to_data = "/Users/Slaton/Documents/Titans/NLP/data.txt"
    data_pd = read_data(path_to_data)
    clean_data = data_pd.copy()
    clean_data["Sequence"] = clean_data["Sequence"].apply(regex_sentence)

    #Transform sentence to vietnamese word-format
    rdrsegmenter = VnCoreNLP("/Users/Slaton/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
    labels = list(np.unique(clean_data["Label"]))
    segmented_dict = data_label_dict(clean_data, rdrsegmenter)
    vocab = set([word for k in segmented_dict.keys() for word in k.split()])  #3938
    final_data = create_pd_dummies_label(segmented_dict, labels)

    #train_test_split
    labels = final_data[[lab for lab in final_data.columns if not lab.startswith("Sequence")]].values
    Xfinal = final_data["Sequence"].values
    X_train, X_val, y_train, y_val = train_test_split(Xfinal, labels, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=0)

    #define weight
    weight = 1 / np.sum(y_train, axis=0)

    #bert processing
    train_inputs_id, train_mask = bert_preprocessing(X_train)
    val_inputs_id, val_mask = bert_preprocessing(X_val)
    test_inputs_id, test_mask = bert_preprocessing(X_test)

    #Dataloader
    train_lab = torch.tensor(y_train, dtype=torch.float)
    val_lab = torch.tensor(y_val, dtype=torch.float)
    test_lab = torch.tensor(y_test, dtype=torch.float)

    train_data = TensorDataset(train_inputs_id, train_mask, train_lab)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_data = TensorDataset(val_inputs_id, val_mask, val_lab)
    val_sampler = RandomSampler(val_data)
    val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    test_data = TensorDataset(test_inputs_id, test_mask, test_lab)
    test_sampler = RandomSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    #model, optimizer, scheduler
    total_step = len(train_loader) * EPOCHS
    bert_classifier, optimizer, scheduler = initialize_model(total_step, EPOCHS)

    #Training
    train(bert_classifier, optimizer,scheduler, train_loader, val_loader, weight)

