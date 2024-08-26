'''
We have one csv file for each patient under `ROOT/data/HuProt_csv_by_patient/`.
We also have an excel file containing protein ID to sequence mapping and protein information.

This script constructs a summary csv that contains the sequence-specific category-specific HuProt scores.
'''

import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm


if __name__ == '__main__':

    per_patient_csv_list = glob('../../data/HuProt_csv_by_patient/*.csv')
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

    # 2. Find all unique protein JHU IDs in per-patient csv files.
    all_protein_JHU_ID_set = set()
    missing_JHU_ID_set = set()
    available_JHU_ID_set = set()
    for csv_path in tqdm(per_patient_csv_list, desc='Finding all proteins in per-patient csv files'):
        df = pd.read_csv(csv_path)
        for item in df['ID'].values:
            protein_id = item.split('.')[0]
            if not protein_id[:3] == 'JHU':
                continue

            all_protein_JHU_ID_set.add(protein_id)

            if protein_id not in JHU_ID_to_sequence_map.keys():
                missing_JHU_ID_set.add(protein_id)
            else:
                available_JHU_ID_set.add(protein_id)

    available_protein_JHU_IDs = np.unique(list(available_JHU_ID_set))

    # Create the summary dataframe.
    df_summary = pd.DataFrame({
        'JHU ID': available_protein_JHU_IDs,
        'Sequence': np.full(len(available_protein_JHU_IDs), np.nan),
        'HuProt_all': np.full(len(available_protein_JHU_IDs), np.nan),
        'HuProt_LC': np.full(len(available_protein_JHU_IDs), np.nan),
        'HuProt_HC': np.full(len(available_protein_JHU_IDs), np.nan),
        'HuProt_CVC': np.full(len(available_protein_JHU_IDs), np.nan),
    })

    # 3. Populate the HuProt scores (overall and by category).
    # LC: long covid
    # HC: healthy control
    # CVC: covalence control
    JHU_ID_to_HuProt_map = {}
    for csv_path in tqdm(per_patient_csv_list, desc='Populating HuProt Scores'):
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

        # This is the HuProt scores of many proteins for this patient.
        df_curr_patient = pd.read_csv(csv_path)
        all_JHU_IDs_in_df_set = set([item for item in df_summary['JHU ID'].values])

        for idx, row in df_curr_patient.iterrows():
            protein_id = row['ID'].split('.')[0]
            if protein_id[:3] != 'JHU':
                continue
            if protein_id not in all_JHU_IDs_in_df_set:
                continue

            # Compute HuProt score.
            huprot_score = row['F635'] - row['B635']

            if protein_id not in JHU_ID_to_HuProt_map:
                JHU_ID_to_HuProt_map[protein_id] = {}
                JHU_ID_to_HuProt_map[protein_id]['all'] = []
                JHU_ID_to_HuProt_map[protein_id]['LC'] = []
                JHU_ID_to_HuProt_map[protein_id]['HC'] = []
                JHU_ID_to_HuProt_map[protein_id]['CVC'] = []

            JHU_ID_to_HuProt_map[protein_id]['all'].append(huprot_score)
            JHU_ID_to_HuProt_map[protein_id][patient_type].append(huprot_score)

    mean_all, mean_LC, mean_HC, mean_CVC = 0, 0, 0, 0
    median_all, median_LC, median_HC, median_CVC = 0, 0, 0, 0
    pctl_75_all, pctl_75_LC, pctl_75_HC, pctl_75_CVC = 0, 0, 0, 0
    pctl_90_all, pctl_90_LC, pctl_90_HC, pctl_90_CVC = 0, 0, 0, 0
    pctl_95_all, pctl_95_LC, pctl_95_HC, pctl_95_CVC = 0, 0, 0, 0
    pctl_99_all, pctl_99_LC, pctl_99_HC, pctl_99_CVC = 0, 0, 0, 0
    max_all, max_LC, max_HC, max_CVC = 0, 0, 0, 0

    # Populate these HuProt Scores to `df_summary`.
    # NOTE: Using the 99th percentile for HuProt score.
    for idx, row in df_summary.iterrows():
        protein_id = row['JHU ID'].split('.')[0]
        if protein_id[:3] != 'JHU':
            continue

        assert protein_id in JHU_ID_to_sequence_map.keys()
        df_summary.loc[idx, 'Sequence'] = JHU_ID_to_sequence_map[protein_id]

        if protein_id in JHU_ID_to_HuProt_map.keys():
            assert len(JHU_ID_to_HuProt_map[protein_id]['all']) > 0

            mean_all += np.mean(JHU_ID_to_HuProt_map[protein_id]['all']) > 1000
            median_all += np.median(JHU_ID_to_HuProt_map[protein_id]['all']) > 1000
            pctl_75_all += np.percentile(JHU_ID_to_HuProt_map[protein_id]['all'], 75) > 1000
            pctl_90_all += np.percentile(JHU_ID_to_HuProt_map[protein_id]['all'], 90) > 1000
            pctl_95_all += np.percentile(JHU_ID_to_HuProt_map[protein_id]['all'], 95) > 1000
            pctl_99_all += np.percentile(JHU_ID_to_HuProt_map[protein_id]['all'], 99) > 1000
            max_all += np.max(JHU_ID_to_HuProt_map[protein_id]['all']) > 1000

            df_summary.loc[idx, 'HuProt_all'] = np.percentile(JHU_ID_to_HuProt_map[protein_id]['all'], 99)

            if len(JHU_ID_to_HuProt_map[protein_id]['LC']) > 0:
                mean_LC += np.mean(JHU_ID_to_HuProt_map[protein_id]['LC']) > 1000
                median_LC += np.median(JHU_ID_to_HuProt_map[protein_id]['LC']) > 1000
                pctl_75_LC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['LC'], 75) > 1000
                pctl_90_LC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['LC'], 90) > 1000
                pctl_95_LC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['LC'], 95) > 1000
                pctl_99_LC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['LC'], 99) > 1000
                max_LC += np.max(JHU_ID_to_HuProt_map[protein_id]['LC']) > 1000

                df_summary.loc[idx, 'HuProt_LC'] = np.percentile(JHU_ID_to_HuProt_map[protein_id]['LC'], 99)

            if len(JHU_ID_to_HuProt_map[protein_id]['HC']) > 0:
                mean_HC += np.mean(JHU_ID_to_HuProt_map[protein_id]['HC']) > 1000
                median_HC += np.median(JHU_ID_to_HuProt_map[protein_id]['HC']) > 1000
                pctl_75_HC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['HC'], 75) > 1000
                pctl_90_HC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['HC'], 90) > 1000
                pctl_95_HC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['HC'], 95) > 1000
                pctl_99_HC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['HC'], 99) > 1000
                max_HC += np.max(JHU_ID_to_HuProt_map[protein_id]['HC']) > 1000

                df_summary.loc[idx, 'HuProt_HC'] = np.percentile(JHU_ID_to_HuProt_map[protein_id]['HC'], 99)

            if len(JHU_ID_to_HuProt_map[protein_id]['CVC']) > 0:
                mean_CVC += np.mean(JHU_ID_to_HuProt_map[protein_id]['CVC']) > 1000
                median_CVC += np.median(JHU_ID_to_HuProt_map[protein_id]['CVC']) > 1000
                pctl_75_CVC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['CVC'], 75) > 1000
                pctl_90_CVC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['CVC'], 90) > 1000
                pctl_95_CVC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['CVC'], 95) > 1000
                pctl_99_CVC += np.percentile(JHU_ID_to_HuProt_map[protein_id]['CVC'], 99) > 1000
                max_CVC += np.max(JHU_ID_to_HuProt_map[protein_id]['CVC']) > 1000

                df_summary.loc[idx, 'HuProt_CVC'] = np.percentile(JHU_ID_to_HuProt_map[protein_id]['CVC'], 99)

    print('Mean: ', mean_all, mean_LC, mean_HC, mean_CVC)
    print('Median: ', median_all, median_LC, median_HC, median_CVC)
    print('75%: ', pctl_75_all, pctl_75_LC, pctl_75_HC, pctl_75_CVC)
    print('90%: ', pctl_90_all, pctl_90_LC, pctl_90_HC, pctl_90_CVC)
    print('95%: ', pctl_95_all, pctl_95_LC, pctl_95_HC, pctl_95_CVC)
    print('99%: ', pctl_99_all, pctl_99_LC, pctl_99_HC, pctl_99_CVC)
    print('Max: ', max_all, max_LC, max_HC, max_CVC)

    # Display the unique letters for the protein sequences.
    unique_letters = set()
    for seq in df_summary['Sequence']:
        for letter in seq:
            if letter not in unique_letters:
                unique_letters.add(letter)
    print(unique_letters)

    # Export to csv file.
    df_summary.to_csv(output_csv_path, index=False)

