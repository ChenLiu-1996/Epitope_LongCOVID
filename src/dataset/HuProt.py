from typing import Tuple, Literal
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


data_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])

class HuProtDataset(Dataset):
    def __init__(self,
                 data_csv: str = data_dir + '/data/HuProt_summary/HuProt_summary.csv',
                 subset: Literal['All', 'LC', 'CVC', 'HC'] = None,
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

        sequences = df_gene_protein['Sequence'].to_numpy()
        if self.subset is None:
            HuProt_scores = df_gene_protein[['HuProt_LC', 'HuProt_HC', 'HuProt_CVC']].to_numpy()
        elif self.subset == 'All':
            HuProt_scores = df_gene_protein['HuProt_all'].to_numpy()
        elif self.subset == 'LC':
            HuProt_scores = df_gene_protein['HuProt_LC'].to_numpy()
        elif self.subset == 'HC':
            HuProt_scores = df_gene_protein['HuProt_HC'].to_numpy()
        elif self.subset == 'CVC':
            HuProt_scores = df_gene_protein['HuProt_CVC'].to_numpy()
        else:
            raise ValueError(
                f'HuProtDataset: `subset` has to be one of `All`, `LC`, `HC`, `CVC` or None. Got {self.subset} instead.')

        assert sequences.shape[0] == HuProt_scores.shape[0]

        # Remove the entries with NaN HuProt scores.
        nan_indices = np.argwhere(np.isnan(HuProt_scores))
        if len(nan_indices) > 0:
            assert len(nan_indices.shape) == 2
            assert nan_indices.shape[1] == 1
            nan_indices = nan_indices.flatten()

            mask = np.ones(len(sequences), dtype=bool)
            mask[nan_indices] = False

            sequences = sequences[mask]
            HuProt_scores = HuProt_scores[mask]

        if self.classification:
            raise NotImplementedError()

        self.sequences = sequences
        self.HuProt_scores = HuProt_scores

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx) -> Tuple:
        sequence = self.sequences[idx]
        HuProt_score = self.HuProt_scores[idx]

        return sequence, HuProt_score


if __name__ == '__main__':
    dataset = HuProtDataset()
