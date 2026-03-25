import torch.nn as nn
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup

MODEL_NAME = "vinai/phobert-base"


class PhoBertClassifier(nn.Module):
    def __init__(self, num_classes=11, freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        self.classifier = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.sigmoid = nn.Sigmoid()

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output[0][:, 0, :]
        logits = self.classifier(cls_output)
        return self.sigmoid(logits)


def initialize_model(total_steps):
    model = PhoBertClassifier()
    optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    return model, optimizer, scheduler
