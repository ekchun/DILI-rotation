# DILI

## rotation fall 2025
Most of this is from Amanda's repo, but with additional messy code. New work includes:
- additional ROS and MMP training data
- creating a 'matrix' of cell health values for DILI compounds

### `preprocessing_data`
In this folder is a messy collection of raw and processed data. I performed manual curation and inspection of curves to further filter bad curves (e.g. curves that had a large hill slope or only one concentration above the 50% mark). The `validate_datasets.ipynb` notebook contains the code used to filter and visualize the data.

### `datasets`
This folder contains the final curated data in classification form. The `all_hits` files contain additional information from the curation process. The `dn_class` or `up_class` files are ready for modeling. `DILIRankST_smiles.csv` contains the SMILES and compound annotations for DILIRank and DILIst datasets. The `DILIst Classification` column is the gold standard target variable.

## `chemberta`
Code from Kien Nguyen (UDel) to use ChemBERTa to predict Cell Count classification. 
