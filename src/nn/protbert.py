import re
import torch
import numpy as np
from transformers import BertConfig, BertModel, BertTokenizer


class Regressor(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        '''
        NOTE: can't do BatchNorm because the batch size is 1.
        Batch size is 1 because the input sequence has variable length.
        '''

        # self.fc1 = torch.nn.Linear(latent_dim, latent_dim)
        # self.norm1 = torch.nn.LayerNorm(latent_dim)
        # self.fc2 = torch.nn.Linear(latent_dim, int(np.sqrt(latent_dim)))
        # self.norm2 = torch.nn.LayerNorm(int(np.sqrt(latent_dim)))
        # self.fc3 = torch.nn.Linear(int(np.sqrt(latent_dim)), 1)
        # self.nonlin = torch.nn.LeakyReLU()
        self.fc = torch.nn.Linear(latent_dim, 1)

    def forward(self, z):
        # h = self.norm1(self.nonlin(self.fc1(z)))
        # h = self.norm2(self.nonlin(self.fc2(h)))
        # y_pred = self.fc3(h)
        y_pred = self.fc(z)
        return y_pred


class EncoderRegressor(torch.nn.Module):
    def __init__(self, encoder, regressor):
        super().__init__()

        self.encoder = encoder
        self.regressor = regressor

    def forward(self, *args, **kwargs):
        z = self.encoder(*args, **kwargs).pooler_output
        y_pred = self.regressor(z)
        return y_pred


class ProtBERT(torch.nn.Module):
    def __init__(self,
                 latent_dim: int = 1024,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device

        bert_config = BertConfig.from_pretrained("Rostlab/prot_bert", num_labels=1)
        encoder = BertModel(bert_config)
        regressor = Regressor(latent_dim=latent_dim)

        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = EncoderRegressor(encoder, regressor)


    def forward(self, seq):
        # Format the protein sequence.
        seq = re.sub(r'[UZOB]', 'X', seq)
        seq = ' '.join(seq)

        encoded_input = self.tokenizer(seq, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)

        y_pred_logit = self.model(**encoded_input)
        return y_pred_logit

    def predict_prob(self, seq):
        y_pred_logit = self.forward(seq)
        y_pred_prob = torch.sigmoid(y_pred_logit)
        return y_pred_prob