# scVGATAE
a variational graph attentional autoencoder model for clustering single-cell RNA-seq data
# About
This repository contains the source code for scVGATAE
# Installation
Use python virutal environment with conda
```
conda create -n scvgataeEnv python=3.8 pip
conda activate scvgataeEnv
pip install -r requirements.txt
```
# Datasets
Biase,Baron and Goolam
# Quick Start
The scVGATAE accepts normal scRNA-seq data format: CSV ,TXT,TSV,H5 and 10X
# CSV format
`python scVGATAE.py --dataset-str 'Baron' --dataset 'data.csv' --data-type 'csv'`
# 10X format
`python scVGATAE.py --dataset-str 'PBMC6k' --dataset 'filtered_matrices_mex/hg19' --data-type 'mtx'`
# TXT format
```
python scVGATAE.py \
--dataset-str 'Adipose'\ 
--dataset 'data.txt' \
--data-type 'txt'\
--normalize False\
--sil-stp False
```
# CSV format
`python scVGATAE.py --dataset-str 'Baron' --dataset 'data.csv' --data-type 'csv'`

# Visualization
We provide the visualization code.

`git clone https://github.com/duddubududu/scVGATAE.git`

