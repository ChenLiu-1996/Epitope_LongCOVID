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
        # seq_tokens = encoded_input.input_ids.clone()

        # if self.mask_ratio > 0:
        #     num_to_mask = int(np.ceil(self.mask_ratio * seq_tokens.size(-1)))
        #     eligible_indices = np.arange(1, seq_tokens.size(-1) - 1)  # Avoid [CLS] and [SEP]
        #     masked_indices = np.random.choice(eligible_indices, size=num_to_mask, replace=False)
        #     for idx in masked_indices:
        #         # Replace token with [MASK] token id
        #         encoded_input.input_ids[0, idx] = self.tokenizer.mask_token_id

        outputs = self.encoder_classifier.esm(**encoded_input)
        hidden_state = outputs.last_hidden_state

        # NOTE: Prediction head.
        y_pred_logit = self.encoder_classifier.classifier(hidden_state)

        return y_pred_logit

    def predict_prob(self, seq):
        y_pred_logit = self.forward(seq)
        y_pred_prob = torch.sigmoid(y_pred_logit)
        return y_pred_prob
