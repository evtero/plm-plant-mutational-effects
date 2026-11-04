# Evaluating protein language models approaches for mutational effect prediction in plants

This repository contains the code of the paper *Evaluating protein language models approaches for mutational effect prediction in plants* (Pardo-Otero E, Bolón-Canedo V, and Piñeiro R).


```
data/                  dataset, dataset protein families and the 5 homology-controlled splits
scripts/
  preprocess_data/     annotations and homology-controlled splits
  plm_emb/             training pipeline of supervised models using ESM-1v embeddings and sift scores (see readme.txt)
  plm_zs/              ESM-1v zero-shot
  classic_methods/     SIFT and PROVEAN
  analysis/            predictions collection and figures
```
