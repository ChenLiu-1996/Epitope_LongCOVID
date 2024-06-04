from typing import Tuple, Literal
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


data_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])

class HuProtDataset(Dataset):
    def __init__(self,
                 data_csv: str = data_dir + '/data/HuProt_summary/HuProt_summary.csv',
                 subset: Literal['LC', 'CVC', 'HC'] = None,
                 classification: bool = False):
        '''
        HuProt Score Prediction from sequence.
        '''

        self.data_csv = data_csv
        self.subset = subset
        self.classification = classification

        self._prepare_data()

    def _prepare_data(self):
        df_gene_protein = pd.read_csv(self.data_csv)

        symbols = df_gene_protein['Approved symbol'].tolist()
        sequences = df_gene_protein['Sequence'].tolist()
        if self.subset is None:
            HuProt_scores = df_gene_protein['HuProt_all'].tolist()
        elif self.subset == 'LC':
            HuProt_scores = df_gene_protein['HuProt_LC'].tolist()
        elif self.subset == 'HC':
            HuProt_scores = df_gene_protein['HuProt_HC'].tolist()
        elif self.subset == 'CVC':
            HuProt_scores = df_gene_protein['HuProt_CVC'].tolist()
        else:
            raise ValueError(
                'HuProtDataset: `subset` has to be one of `LC`, `HC`, `CVC` or None. Got %s instead.' % self.subset)

        symbols = np.array(symbols)
        sequences = np.array(sequences)
        HuProt_scores = np.array(HuProt_scores)

        assert symbols.shape == sequences.shape and symbols.shape == HuProt_scores.shape

        # Remove the entries with NaN HuProt scores.
        nan_indices = np.argwhere(np.isnan(HuProt_scores))
        if len(nan_indices) > 0:
            assert len(nan_indices.shape) == 2
            assert nan_indices.shape[1] == 1
            nan_indices = nan_indices.flatten()

            mask = np.ones(symbols.size, dtype=bool)
            mask[nan_indices] = False

            symbols = symbols[mask]
            sequences = sequences[mask]
            HuProt_scores = HuProt_scores[mask]

        if self.classification:
            HuProt_scores = np.uint8(np.array(HuProt_scores) > 1000.0)
        else:
            HuProt_scores = np.array(HuProt_scores)

        self.symbols = symbols
        self.sequences = sequences
        self.HuProt_scores = HuProt_scores

    def __len__(self) -> int:
        return len(self.symbols)

    def __getitem__(self, idx) -> Tuple:
        symbol = self.symbols[idx]
        sequence = self.sequences[idx]
        HuProt_score = self.HuProt_scores[idx]

        return symbol, sequence, HuProt_score
