import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

loss_fn = nn.BCELoss(reduction="none")


def train(model, optimizer, scheduler, train_dataloader, val_dataloader, weight, epochs=4, evaluation=False):
    print("Start training...\n")
    for epoch_i in range(epochs):
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | "
            f"{'Val Loss':^10} | {'Val F1':^9} | {'Val Precision':^9} | {'Val Recall':^9}"
        )

        t0_batch = time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_counts += 1
            b_input_id, b_att_mask, b_label = batch

            output = model(b_input_id, b_att_mask)
            tmp_loss = loss_fn(output, b_label)
            weighted_loss = tmp_loss * torch.tensor(weight)
            loss = weighted_loss.mean()

            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | "
                    f"{'-':^10} | {'-':^9} | {'-':^10} | {'-':^10}"
                )
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        if evaluation:
            val_loss, val_f1, val_precision, val_recall = evaluate(model, val_dataloader, weight)
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | "
                f"{val_loss:^10.6f} | {val_f1:^10.6f} | {val_precision:^10.6f} | {val_recall:^10.6f}"
            )
        print()

    print("Training complete!")


def evaluate(model, val_dataloader, weight):
    model.eval()
    val_loss = []
    val_f1, val_prec, val_recall = [], [], []

    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = batch

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        tmp_loss = loss_fn(logits, b_labels)
        weighted_loss = tmp_loss * torch.tensor(weight)
        loss = weighted_loss.mean()
        val_loss.append(loss.item())

        preds = np.array(logits > 0.5, dtype=float)
        precision = precision_score(b_labels.numpy(), preds, average="weighted", zero_division=0)
        f1 = f1_score(b_labels.numpy(), preds, average="weighted", zero_division=0)
        recall = recall_score(b_labels.numpy(), preds, average="weighted", zero_division=0)

        val_f1.append(f1)
        val_recall.append(recall)
        val_prec.append(precision)

    return np.mean(val_loss), np.mean(val_f1), np.mean(val_prec), np.mean(val_recall)
