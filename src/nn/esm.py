import re
import torch
import numpy as np
from transformers import EsmForSequenceClassification, AutoTokenizer


class ESM(torch.nn.Module):
    def __init__(self,
                #  mask_ratio: float = 0.15,
                 num_esm_layers: int = None,
                #  max_seq_len: int = 2048,
                 device: torch.device = torch.device('cpu')):

        super().__init__()
        # self.mask_ratio = mask_ratio
        self.device = device
        self.num_esm_layers = num_esm_layers
        # self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.encoder_classifier = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=1)

        if self.num_esm_layers is not None:
            assert self.num_esm_layers > 0 and self.num_esm_layers <= len(self.encoder_classifier.esm.encoder.layer)
            self.encoder_classifier.esm.encoder.layer = self.encoder_classifier.esm.encoder.layer[:self.num_esm_layers]

    def forward(self, seq):
        # Format the protein sequence.
        seq = re.sub(r'[UZOB]', 'X', seq)
        seq = ' '.join(seq)

        encoded_input = self.tokenizer(seq, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)

        outputs = self.encoder_classifier.esm(**encoded_input)
        hidden_state = outputs.last_hidden_state

        # NOTE: Prediction head.
        y_pred_logit = self.encoder_classifier.classifier(hidden_state)

        return y_pred_logit

    def output_attentions(self, seq):
        # Format the protein sequence.
        seq = re.sub(r'[UZOB]', 'X', seq)
        seq = ' '.join(seq)

        encoded_input = self.tokenizer(seq, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)

        outputs = self.encoder_classifier.esm(**encoded_input, output_attentions=True)
        hidden_state = outputs.last_hidden_state
        attentions = outputs.attentions

        # NOTE: Prediction head.
        y_pred_logit = self.encoder_classifier.classifier(hidden_state)

        return y_pred_logit, attentions

    def predict_prob(self, seq):
        y_pred_logit = self.forward(seq)
        y_pred_prob = torch.sigmoid(y_pred_logit)
        return y_pred_prob
