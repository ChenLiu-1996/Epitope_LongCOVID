import re
import torch
import numpy as np
from transformers import EsmForSequenceClassification, AutoTokenizer, EsmForMaskedLM


class ReconESM(torch.nn.Module):
    def __init__(self,
                 mask_ratio: float = 0.15,
                 num_esm_layers: int = None,
                 max_seq_len: int = 2048,
                 device: torch.device = torch.device('cpu')):

        super().__init__()
        self.mask_ratio = mask_ratio
        self.device = device
        self.num_esm_layers = num_esm_layers
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.encoder_classifier = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=1)
        model_mlm = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.reconstruction_head = model_mlm.lm_head

        self.latent_dim = self.encoder_classifier.classifier.dense.in_features
        self.global_attn_module = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 1), torch.nn.Softmax(dim=1)
        )

        # self.decoder = AutoregressiveTransformer(vocab_size=self.tokenizer.vocab_size,
        #                                          embed_size=self.latent_dim,
        #                                          num_heads=4,
        #                                          num_layers=2)

        self.decoder = ConvDecoder(latent_dim=self.latent_dim,
                                   max_seq_len=self.max_seq_len,
                                   vocab_size=self.tokenizer.vocab_size)

        if self.num_esm_layers is not None:
            assert self.num_esm_layers > 0 and self.num_esm_layers <= len(self.encoder_classifier.esm.encoder.layer)
            self.encoder_classifier.esm.encoder.layer = self.encoder_classifier.esm.encoder.layer[:self.num_esm_layers]

        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(self.latent_dim, self.latent_dim),
        #     torch.nn.LayerNorm(self.latent_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.latent_dim, self.max_seq_len * self.tokenizer.vocab_size),
        # )

    def forward(self, seq):
        # Format the protein sequence.
        seq = re.sub(r'[UZOB]', 'X', seq)
        seq = ' '.join(seq)

        if len(seq) > self.max_seq_len:
            print('\n\nWTF? seq len exceeded?')
            print('max seq len:', self.max_seq_len)
            print('seq len:', len(seq))

        encoded_input = self.tokenizer(seq, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)
        seq_tokens = encoded_input.input_ids.clone()

        if self.mask_ratio > 0:
            num_to_mask = int(np.ceil(self.mask_ratio * seq_tokens.size(-1)))
            eligible_indices = np.arange(1, seq_tokens.size(-1) - 1)  # Avoid [CLS] and [SEP]
            masked_indices = np.random.choice(eligible_indices, size=num_to_mask, replace=False)
            for idx in masked_indices:
                # Replace token with [MASK] token id
                encoded_input.input_ids[0, idx] = self.tokenizer.mask_token_id

        outputs = self.encoder_classifier.esm(**encoded_input)
        hidden_state = outputs.last_hidden_state

        # NOTE: Prediction head.
        y_pred_logit = self.encoder_classifier.classifier(hidden_state)

        # NOTE: Latent embedding at bottleneck.
        glob_attn = self.global_attn_module(hidden_state)  # glob_attn: [batch_size, seq_len, 1]
        latent_emb = torch.bmm(glob_attn.transpose(-1, 1), hidden_state).squeeze()
        # Regain the batch dimension.
        if len(hidden_state) == 1:
            latent_emb = latent_emb.unsqueeze(0)  # [batch_size, num_feature]

        # NOTE: Reconstruction head.
        # seq_recon_logit = self.reconstruction_head(hidden_state)
        seq_recon_logit = self.decoder(latent_emb)  # [batch_size, seq_len, vocab_size]
        curr_seq_len = seq_tokens.shape[1]
        seq_recon_logit = seq_recon_logit[:, :curr_seq_len, :]

        # Permute `seq_recon_logit` to match expected input shape for CrossEntropyLoss
        seq_recon_logit = seq_recon_logit.permute(0, 2, 1)  # [batch_size, vocab_size, seq_len]

        return y_pred_logit, seq_tokens, seq_recon_logit, latent_emb

    def predict_prob(self, seq):
        y_pred_logit = self.forward(seq)
        y_pred_prob = torch.sigmoid(y_pred_logit)
        return y_pred_prob


class ConvDecoder(torch.nn.Module):
    def __init__(self, latent_dim, max_seq_len, vocab_size, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.mapper = torch.nn.Linear(self.latent_dim, self.max_seq_len * self.hidden_dim)
        self.hidden_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            torch.nn.InstanceNorm1d(self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            torch.nn.InstanceNorm1d(self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Conv1d(self.hidden_dim, self.vocab_size, kernel_size=3, padding=1)

    def forward(self, z):
        # z: [batch_size, num_feature]
        z = self.mapper(z) # [batch_size, seq_len x hidden_dim]
        z = z.reshape(-1, self.hidden_dim, self.max_seq_len) # [batch_size, hidden_dim, seq_len]
        z_residual = self.hidden_layers(z) # [batch_size, hidden_dim, seq_len]
        z = z + z_residual # [batch_size, hidden_dim, seq_len]
        z = self.output_layer(z) # [batch_size, vocab_size, seq_len]
        z = torch.transpose(z, 1, 2) # [batch_size, seq_len, vocab_size]
        return z

# class AutoregressiveTransformer(torch.nn.Module):
#     def __init__(self, vocab_size, embed_size, num_heads, num_layers):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, embed_size)
#         self.transformer = torch.nn.Transformer(embed_size, num_heads, num_layers)
#         self.fc_out = torch.nn.Linear(embed_size, vocab_size)

#     def forward(self, x):
#         import pdb; pdb.set_trace()
#         # Create the mask to ensure autoregression
#         batch_size, seq_len = x.shape
#         mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)  # Lower triangular matrix

#         # Embed the input sequence
#         embedded = self.embedding(x)

#         # Apply the transformer with the causal mask
#         transformer_out = self.transformer(embedded, embedded, src_mask=mask)

#         # Output layer to map to vocab size
#         output = self.fc_out(transformer_out)
#         return output