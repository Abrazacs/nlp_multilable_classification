from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=None):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size, num_labels or config.num_labels)
        self.num_labels = num_labels or config.num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}