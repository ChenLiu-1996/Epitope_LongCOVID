'''
We have a csv file under `ROOT/data/HuProt_batch_corrected/`.
We also have an excel file containing protein ID to sequence mapping and protein information.

This script constructs a summary csv that contains
1. The sequence-specific patient-specific HuProt scores.
2. The sequence-specific category-specific HuProt scores.
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':

    input_csv_path = '../../data/HuProt_batch_corrected/TP_HuProt_b1b2_Lg2CPM_TMM_avg_LCIDbFull.csv'
    sequence_mapping_excel_path = '../../data/Pilot_Raw_IgG.xlsx'

    output_csv_path = '../../data/HuProt_summary/HuProt_summary.csv'
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Create a unified dataframe with all information.
    # 1. Find the sequene for each protein.
    JHU_ID_to_sequence_map = {}
    df_protein_info = pd.read_excel(sequence_mapping_excel_path, sheet_name="HuProt_HPA")
    for idx, row in tqdm(df_protein_info.iterrows(), desc='Finding the sequence for each protein', total=len(df_protein_info)):
        protein_id = row['Clone']
        assert protein_id[:3] == 'JHU'
        sequence = row['Sequence']
        assert protein_id not in JHU_ID_to_sequence_map
        JHU_ID_to_sequence_map[protein_id] = sequence
    del protein_id

    # 2. Find all unique protein JHU IDs in the batch corrected csv file.
    all_protein_id_set = set()
    missing_protein_id_set = set()
    available_protein_id_set = set()
    huprot_score_map = {}
    df_input = pd.read_csv(input_csv_path)
    # Proteins are the columns.
    for protein_id in tqdm(df_input.columns, desc='Filtering proteins without sequence information'):
        if 'JHU' not in protein_id:
            continue
        protein_JHUID = 'JHU' + protein_id.split('JHU')[1]

        all_protein_id_set.add(protein_id)
        if protein_JHUID not in JHU_ID_to_sequence_map.keys():
            missing_protein_id_set.add(protein_id)
        else:
            available_protein_id_set.add(protein_id)
            huprot_score_map[protein_id] = {
                'all': [],
                'HC': [],
                'CVC': [],
                'LC': [],
            }
    del protein_id, protein_JHUID

    # 3. Populate the HuProt scores (overall, by category, and by patient).
    # LC: long covid
    # HC: healthy control
    # CVC: covalence control
    for idx, row in tqdm(df_input.iterrows(), desc='Populating HuProt scores', total=len(df_input)):
        patient = row['LCID_Batch']
        patient_is_LC = 'LC' in patient
        patient_is_HC = 'HC' in patient
        patient_is_CVC = 'CVC' in patient
        assert patient_is_LC + patient_is_HC + patient_is_CVC == 1
        if patient_is_LC:
            patient_type = 'LC'
        elif patient_is_HC:
            patient_type = 'HC'
        elif patient_is_CVC:
            patient_type = 'CVC'

        for protein_id in huprot_score_map.keys():
            # Find HuProt score.
            huprot_score = row[protein_id]
            huprot_score_map[protein_id]['all'].append(huprot_score)
            huprot_score_map[protein_id][patient_type].append(huprot_score)
            huprot_score_map[protein_id][patient] = huprot_score
    del patient, protein_id

    # 4. Populate these HuProt Scores to `df_summary`.
    # Create the summary dataframe.
    available_protein_ids = np.unique(list(available_protein_id_set))
    all_patients = df_input['LCID_Batch'].values.tolist()
    df_summary_map = {
        'protein_id': available_protein_ids,
        'Sequence': np.full(len(available_protein_ids), np.nan),
        'HuProt_all': np.full(len(available_protein_ids), np.nan),
        'HuProt_LC': np.full(len(available_protein_ids), np.nan),
        'HuProt_HC': np.full(len(available_protein_ids), np.nan),
        'HuProt_CVC': np.full(len(available_protein_ids), np.nan),
    }
    for patient in all_patients:
        df_summary_map[patient] = np.full(len(available_protein_ids), np.nan)
    df_summary = pd.DataFrame(df_summary_map)
    del df_summary_map

    # NOTE: Using the median for HuProt score.
    for idx, row in tqdm(df_summary.iterrows(), desc='Creating summary csv', total=len(df_summary)):
        protein_id = row['protein_id']
        protein_JHUID = 'JHU' + protein_id.split('JHU')[1]

        assert protein_JHUID in JHU_ID_to_sequence_map.keys()
        df_summary.loc[idx, 'Sequence'] = JHU_ID_to_sequence_map[protein_JHUID]

        for patient in all_patients:
            # df_summary.loc[idx, patient] = df_input.loc[df_input['LCID_Batch'] == patient, protein_id].item()
            df_summary.loc[idx, patient] = huprot_score_map[protein_id][patient]

        if protein_id in huprot_score_map.keys():
            assert len(huprot_score_map[protein_id]['all']) > 0
            df_summary.loc[idx, 'HuProt_all'] = np.median(huprot_score_map[protein_id]['all'])

            if len(huprot_score_map[protein_id]['LC']) > 0:
                df_summary.loc[idx, 'HuProt_LC'] = np.median(huprot_score_map[protein_id]['LC'])

            if len(huprot_score_map[protein_id]['HC']) > 0:
                df_summary.loc[idx, 'HuProt_HC'] = np.median(huprot_score_map[protein_id]['HC'])

            if len(huprot_score_map[protein_id]['CVC']) > 0:
                df_summary.loc[idx, 'HuProt_CVC'] = np.median(huprot_score_map[protein_id]['CVC'])

    # Display the unique letters for the protein sequences.
    unique_letters = set()
    for seq in df_summary['Sequence']:
        for letter in seq:
            if letter not in unique_letters:
                unique_letters.add(letter)
    print(unique_letters)

    # Export to csv file.
    df_summary.to_csv(output_csv_path, index=False)

