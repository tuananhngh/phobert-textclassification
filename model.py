import torch.nn as nn
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup


class PhoBertClassifier(nn.Module):
    def __init__(self, freeze=True):
        super(PhoBertClassifier, self).__init__()
        d_in, hidden, d_out = 768, 64, 11
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")

        self.classifier = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out))
        self.sgm = nn.Sigmoid()
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, inp_id, att_msk):
        output = self.bert(input_ids=inp_id, attention_mask=att_msk)
        last_hidden = output[0][:, 0, :]
        output_cls = self.classifier(last_hidden)
        output_sgm = self.sgm(output_cls)
        return output_sgm


def initialize_model(total_step, epoch=4):
    phobert = PhoBertClassifier()
    optimizer = AdamW(phobert.parameters(),
                      lr=3e-5,
                      eps=1e-8)
    #total_steps = len(train_loader)*epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_step)
    return phobert, optimizer, scheduler
