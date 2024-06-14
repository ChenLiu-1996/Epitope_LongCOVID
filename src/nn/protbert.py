import re
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer


class ProtBERT(torch.nn.Module):
    def __init__(self,
                 num_bert_layers: int = None,
                 device: torch.device = torch.device('cpu')):

        super().__init__()
        self.device = device
        self.num_bert_layers = num_bert_layers

        self.model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", num_labels=1)
        if self.num_bert_layers is not None:
            assert self.num_bert_layers > 0 and self.num_bert_layers <= len(self.model.bert.encoder.layer)
            self.model.bert.encoder.layer = self.model.bert.encoder.layer[:self.num_bert_layers]

        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    def forward(self, seq):
        # Format the protein sequence.
        seq = re.sub(r'[UZOB]', 'X', seq)
        seq = ' '.join(seq)

        encoded_input = self.tokenizer(seq, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)

        y_pred_logit = self.model(**encoded_input).logits
        return y_pred_logit

    def predict_prob(self, seq):
        y_pred_logit = self.forward(seq)
        y_pred_prob = torch.sigmoid(y_pred_logit)
        return y_pred_prob