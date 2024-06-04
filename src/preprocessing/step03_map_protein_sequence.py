'''
We have
1) one csv file for each patient under `ROOT/data/HuProt_csv_by_patient/`.
2) a csv file: `input02_gene_names.csv` which contains the gene names, gene ids, and other info.
    This is from genenames.org
3) an Excel file: `input03_protein_info.xlsx` which contains the protein sequences and other info.
    This is from uniprot.org
4) a tsv file: `proteinatlas_search.tsv` which conteins the gene description and protein class info.
    This is from proteinatlas.org
We want to process and merge them into a single csv of HuProt scores, sequences and other info.
'''

import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm


if __name__ == '__main__':

    gene_info_path = './input02_gene_names.csv'
    protein_info_path = './input03_protein_info.xlsx'
    protein_atlas_path = './proteinatlas_search.tsv'
    per_patient_csv_list = glob('../../data/HuProt_csv_by_patient/*.csv')

    output_csv_path = '../../data/HuProt_summary/HuProt_summary.csv'
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df_gene_info = pd.read_csv(gene_info_path)
    df_protein_info = pd.read_excel(protein_info_path)
    df_protein_atlas = pd.read_csv(protein_atlas_path, header=0, sep='\t')

    # Create a unified dataframe with all information.
    # 1. Merge the gene df and the protein info df.
    df_gene_protein = pd.merge(df_gene_info,
                               df_protein_info.rename(columns={'From': 'HGNC ID'}),
                               on='HGNC ID', how='inner')

    # Deduplicate rows. Occasionally we have duplicates due to upper/lower case.
    df_gene_protein = df_gene_protein.drop_duplicates()

    # 2. Populate the HuProt scores (overall and by category).
    # LC: long covid
    # HC: healthy control
    # CVC: covalence control
    row_to_HuProt_map = {}
    for csv_path in tqdm(per_patient_csv_list):
        patient_filename = os.path.basename(csv_path)
        patient_string = patient_filename.split('_')[0]

        patient_is_LC = 'LC' in patient_string
        patient_is_HC = 'HC' in patient_string
        patient_is_CVC = 'CVC' in patient_string
        assert patient_is_LC + patient_is_HC + patient_is_CVC == 1
        if patient_is_LC:
            patient_type = 'LC'
        elif patient_is_HC:
            patient_type = 'HC'
        elif patient_is_CVC:
            patient_type = 'CVC'

        # HuProt scores of many genes for this patient.
        df_curr_patient = pd.read_csv(csv_path)
        df_gene_protein_gene_name_uppercase = np.array([item.upper() for item in df_gene_protein['Input'].values])

        for idx, row in df_curr_patient.iterrows():
            gene_id = row['Name'].upper()
            if gene_id in df_gene_protein_gene_name_uppercase:
                loc_match = np.argwhere(df_gene_protein_gene_name_uppercase == gene_id.upper())

                if not len(loc_match) == 1 and len(loc_match[0]) == 1:
                    # Multiple sequences from the same HGNC ID.
                    # We will ignore them for now.
                    assert len(loc_match) > 1
                    continue

                # Now we are certain that the gene_id appears in `df_gene_protein`.
                huprot_score = row['F635'] - row['B635']

                if gene_id not in row_to_HuProt_map:
                    row_to_HuProt_map[gene_id] = {}
                    row_to_HuProt_map[gene_id]['all'] = []
                    row_to_HuProt_map[gene_id]['LC'] = []
                    row_to_HuProt_map[gene_id]['HC'] = []
                    row_to_HuProt_map[gene_id]['CVC'] = []

                row_to_HuProt_map[gene_id]['all'].append(huprot_score)
                row_to_HuProt_map[gene_id][patient_type].append(huprot_score)

    # Add 4 columns in `df_gene_protein` that record HuProt scores.
    df_gene_protein['HuProt_all'] = np.full(len(df_gene_protein), np.nan)
    df_gene_protein['HuProt_LC'] = np.full(len(df_gene_protein), np.nan)
    df_gene_protein['HuProt_HC'] = np.full(len(df_gene_protein), np.nan)
    df_gene_protein['HuProt_CVC'] = np.full(len(df_gene_protein), np.nan)

    # Populate these HuProt Scores.
    for idx, row in df_gene_protein.iterrows():
        gene_id = row['Approved symbol'].upper()
        if gene_id in row_to_HuProt_map.keys():
            assert len(row_to_HuProt_map[gene_id]['all']) > 0

            # NOTE: Using the 99th percentile for HuProt score.
            df_gene_protein.loc[idx, 'HuProt_all'] = np.percentile(row_to_HuProt_map[gene_id]['all'], 99)
            if len(row_to_HuProt_map[gene_id]['LC']) > 0:
                df_gene_protein.loc[idx, 'HuProt_LC'] = np.percentile(row_to_HuProt_map[gene_id]['LC'], 99)
            if len(row_to_HuProt_map[gene_id]['HC']) > 0:
                df_gene_protein.loc[idx, 'HuProt_HC'] = np.percentile(row_to_HuProt_map[gene_id]['HC'], 99)
            if len(row_to_HuProt_map[gene_id]['CVC']) > 0:
                df_gene_protein.loc[idx, 'HuProt_CVC'] = np.percentile(row_to_HuProt_map[gene_id]['CVC'], 99)


    # 3. Annotate the protein using the protein atlas.
    protein_class_list = []
    for _, row in df_gene_protein.iterrows():
        symbol = row['Approved symbol']
        try:
            protein_class = df_protein_atlas.loc[df_protein_atlas['Gene'] == symbol, 'Protein class'].item()
        except:
            protein_class = ""
        protein_class_list.append(protein_class)
    df_gene_protein['Protein class'] = protein_class_list

    # Display the unique letters for the protein sequences.
    unique_letters = set()
    for seq in df_gene_protein['Sequence']:
        for letter in seq:
            if letter not in unique_letters:
                unique_letters.add(letter)
    print(unique_letters)

    # Export to csv file.
    df_gene_protein.to_csv(output_csv_path, index=False)

