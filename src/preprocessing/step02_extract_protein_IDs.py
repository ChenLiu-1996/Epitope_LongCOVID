'''
We have a csv file: `input02_gene_names.csv`.
We want to extract the HGNC protein IDs.
Then we can download the corresponding protein sequences.

NOTE: If your `.csv` begins with `sep=,` on a single line, you may manually delete that `sep=,` line.
'''

import pandas as pd


if __name__ == '__main__':

    input_csv_path = './input02_gene_names.csv'
    output_protein_id_txt_path = './output02_protein_ids.txt'

    df = pd.read_csv(input_csv_path)

    protein_HGNC_ids = df['HGNC ID'].values
    valid_gene_ids = df['Approved symbol'].values

    # Export to txt file.
    # This is a space-separated filed of protein IDs (HGNC format).
    with open(output_protein_id_txt_path, 'w') as f:
        f.write(' '.join([str(item) for item in protein_HGNC_ids]))


    '''
    # NOTE: Once you obtain the output `output02_protein_ids.txt` file:
    # 1. Copy the content (space-separated strings)
    # 2. Go to https://www.uniprot.org/id-mapping
    # 3. Paste the content to the big search box.
    # 4. Below the search box,
    #    - Under `from database`, select `HGNC`.
    #    - Under `to database`, select `UniProtKB/Swiss-Prot`.
    # 5. Click `Map IDs`.
    # 6. Download the excel into `input03_protein_info.xlsx`.
    # 7. When downloading, use the `Excel` format, select `Compressed: no`,
    #    and ensure these columns are selected.
    #    - UniProt Data > Names & Taxonomy > Entry Name
    #    - UniProt Data > Names & Taxonomy > Gene Names
    #    - UniProt Data > Names & Taxonomy > Organism
    #    - UniProt Data > Names & Taxonomy > Protein names
    #    - UniProt Data > Sequences > Length
    #    - UniProt Data > Sequences > Sequence
    ......
    ......
    # Meanwhile, you can do the following.
    # 1. Go to https://www.proteinatlas.org/humanproteome/proteinclasses
    # 2. Click `Search` with an empty search box, and let it return all results.
    # 3. Click `Download custom TSV/JSON`
    # 4. In `Select data columns`, select `Gene`, `Gene description` and `Protein class`.
    # 5. Click `Download TSV`.
    # 6. Save as `proteinatlas_search.tsv`
    # We can do filtering on our own with `output02_valid_gene_ids.txt` in the next script.
    # We are doing it this way because batched searching with many items is not supported.
    '''
