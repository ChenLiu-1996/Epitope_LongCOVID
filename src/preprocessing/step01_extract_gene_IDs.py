'''
We have one csv file for each patient under `ROOT/data/HuProt_csv_by_patient/`.
We want to extract the gene IDs that we care about.
Then we can find the corresponding proteins and download the protein sequences.
'''

import numpy as np
import pandas as pd
from glob import glob


if __name__ == '__main__':

    per_patient_csv_list = glob('../../data/HuProt_csv_by_patient/*.csv')
    output_txt_path = './output01_gene_IDs.txt'

    # Find the gene IDs.
    all_gene_list = []
    for csv_path in per_patient_csv_list:
        df = pd.read_csv(csv_path)
        for item in df.Name.values:
            all_gene_list.append(item)

    all_gene_ids = np.unique(all_gene_list)

    # Export to txt file.
    # This is a space-separated filed of gene IDs.
    with open(output_txt_path, 'w') as f:
        f.write(' '.join([str(item) for item in all_gene_ids]))

    '''
    # NOTE: Once you obtain the output txt file:
    # 1. Copy the content (space-separated strings)
    # 2. Go to https://www.genenames.org/tools/multi-symbol-checker/
    # 3. Paste the content to the big search box on the left-hand side.
    # 4. On the right-hand-side, de-select
    #   `Withdrawn`, `Unmatched`, `Previous symbols` and `Alias symbols`,
    #   while keeping `Approved symbols`.
    # 5. Click `submit`.
    # 6. Save the csv into `input02_gene_names.csv`.
    '''
