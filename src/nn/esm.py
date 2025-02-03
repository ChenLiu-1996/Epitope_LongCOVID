import re
import torch
import numpy as np
from transformers import EsmForSequenceClassification, AutoTokenizer


class ESM(torch.nn.Module):
    def __init__(self,
                 num_esm_layers: int = None,
                 num_classes: int = 3,
                 device: torch.device = torch.device('cpu')):

        super().__init__()
        self.num_esm_layers = num_esm_layers
        self.num_classes = num_classes
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.encoder_classifier = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=self.num_classes)

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

    def output_attribution(self, seq):
        # Format the protein sequence.
        seq = re.sub(r'[UZOB]', 'X', seq)
        seq = ' '.join(seq)

        encoded_input = self.tokenizer(seq, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)

        outputs = self.encoder_classifier.esm(**encoded_input, output_attentions=True)
        hidden_state = outputs.last_hidden_state  # [1, L, 320]
        attentions = outputs.attentions
        attention_rolled = attention_rollout(attentions)
        token_level_attention = torch.sum(attention_rolled, dim=-1)

        # Select CLS token embedding (first token).
        cls_embedding = hidden_state[:, 0, :].clone().detach().requires_grad_(True) # [1, 320]
        logits = self.encoder_classifier.classifier(cls_embedding[:, None, :])  # [1, num_classes]

        token_attribution = None
        for target_class in range(self.num_classes):
            # Compute gradient of target class score w.r.t. CLS token.
            self.encoder_classifier.zero_grad()
            target_score = logits[:, target_class]
            target_score.backward(retain_graph=True)

            # Retrieve gradients of CLS token embedding.
            cls_grad = cls_embedding.grad

            # Compute per-token gradient using CLS gradients.
            token_gradient = (hidden_state * cls_grad).sum(dim=-1)  # (1, L)

            # Final token importance
            token_importance = token_gradient * token_level_attention
            if token_attribution is None:
                token_attribution = token_importance.cpu().detach().numpy()
            else:
                token_attribution = np.vstack((token_attribution, token_importance.cpu().detach().numpy()))

        return token_attribution

    def predict_prob(self, seq):
        y_pred_logit = self.forward(seq)
        y_pred_prob = torch.sigmoid(y_pred_logit)
        return y_pred_prob


def attention_rollout(attentions, discard_ratio=0.95, head_fusion='max'):
    device = attentions[0].device
    result = torch.eye(attentions[0].size(-1)).to(device)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but don't drop the class token.
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    return result
