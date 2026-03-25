import argparse
import numpy as np
import torch
import random
from vncorenlp import VnCoreNLP
from dataprocessing import read_data, regex_sentence
from segmenting import data_label_dict, create_pd_dummies_label
from model import initialize_model
from trainer import train
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

BATCH_SIZE = 32
EPOCHS = 5
MAX_SEQ_LEN = 114
MODEL_NAME = "vinai/phobert-base"


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


def bert_preprocessing(data, tokenizer):
    inputs_id = []
    att_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        inputs_id.append(encoded_sent.get("input_ids"))
        att_masks.append(encoded_sent.get("attention_mask"))
    inputs_id = torch.tensor(inputs_id)
    att_masks = torch.tensor(att_masks)
    return inputs_id, att_masks


def make_dataloader(inputs_id, mask, labels, batch_size):
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    dataset = TensorDataset(inputs_id, mask, labels_tensor)
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description="PhoBERT Text Classification")
    parser.add_argument("--data", type=str, required=True, help="Path to the input JSON data file")
    parser.add_argument("--vncorenlp", type=str, required=True, help="Path to VnCoreNLP jar file")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Read and clean data
    data_pd = read_data(args.data)
    clean_data = data_pd.copy()
    clean_data["Sequence"] = clean_data["Sequence"].apply(regex_sentence)

    # Segment sentences into Vietnamese word format
    rdrsegmenter = VnCoreNLP(args.vncorenlp, annotators="wseg", max_heap_size="-Xmx500m")
    labels = list(np.unique(clean_data["Label"]))
    segmented_dict = data_label_dict(clean_data, rdrsegmenter)
    final_data = create_pd_dummies_label(segmented_dict, labels)

    # Train/val/test split
    label_cols = [col for col in final_data.columns if col != "Sequence"]
    labels = final_data[label_cols].values
    X = final_data["Sequence"].values
    X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=0)

    # Class weights for imbalanced data
    weight = 1 / np.sum(y_train, axis=0)

    # Tokenize with PhoBERT tokenizer
    tokenizer = get_tokenizer()
    train_inputs_id, train_mask = bert_preprocessing(X_train, tokenizer)
    val_inputs_id, val_mask = bert_preprocessing(X_val, tokenizer)
    test_inputs_id, test_mask = bert_preprocessing(X_test, tokenizer)

    # Build DataLoaders
    train_loader = make_dataloader(train_inputs_id, train_mask, y_train, args.batch_size)
    val_loader = make_dataloader(val_inputs_id, val_mask, y_val, args.batch_size)
    test_loader = make_dataloader(test_inputs_id, test_mask, y_test, args.batch_size)

    # Initialize model, optimizer, scheduler
    total_step = len(train_loader) * args.epochs
    bert_classifier, optimizer, scheduler = initialize_model(total_step)

    # Train
    train(bert_classifier, optimizer, scheduler, train_loader, val_loader, weight, epochs=args.epochs)
