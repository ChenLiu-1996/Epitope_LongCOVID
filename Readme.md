# Epitope Prediction (Long COVID NeuroPASC Project)

# LongCovid NeuroPASC
### Krishnaswamy Lab, Yale University
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/NeuroPASC.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/NeuroPASC/)


## Repository Hierarchy
```
```

## Usage
### Preprocessing.
In this section, we prepare the data for training and analysis.
1. Download the patient-specific HuProt score csv files under `./data/HuProt_csv/`
2. Run the following scripts:
```
cd preprocessing
python step01_extract_gene_IDs.py
python step02_extract_protein_IDs.py
python step03_map_protein_sequence.py
```
**Note:** There are action items to perform in between these scripts. These action items are described **inside** the respective scripts. These are mainly accessing some online databases and downloading the queried results. For each script, you need to complete these action items prior to running the next script.


<!-- ### Misc: ProtBERT embedding.
This is some miscellaneous work.

1. We looked at some pre-trained embeddings and see if it can correlate to REAP scores after neural network regression.
2. We performed graph smoothing on the REAP scores.
3. We used a neural network to predict REAP scores after smoothing from pre-trained embeddings.

```
python plot_ProtBERT_embeddings.py

python train_reg_with_ProtBERT_embeddings.py --lr 1e-3 --wd 1e-4
python train_reg_with_ProtBERT_embeddings.py --lr 1e-3 --wd 1e-4 --plot --run_count 11
``` -->



## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name pasc pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -c anaconda -c conda-forge
conda activate pasc
conda install scikit-learn scikit-image pandas matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge
python -m pip install networkx pytorch-lightning==1.9 wandb gdown phate
python -m pip install openpyxl
python -m pip install transformers==4.18.0
python -m pip install open_clip_torch==2.23.0 transformers==4.35.2

python -m pip install datasets
python -m pip install evaluate

python -m pip install bertviz
python -m pip install jupyterlab
python -m pip install ipywidgets


# If it says "version `GLIBCXX_3.4.29' not found", you can try:
`export LD_LIBRARY_PATH=/gpfs/gibbs/pi/krishnaswamy_smita/cl2482/.conda_envs/pasc/lib/:$LD_LIBRARY_PATH`
Replace the path with your conda env path.

```

