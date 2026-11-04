TRAINING PIPELINE
=================

ESM-1v embeddings (+ optional SIFT score) -> ML classifiers (SVC, Random Forest, XGBoost)
to predict functional vs neutral missense variants in plants.

main.py runs, for each of the 5 family-level splits:
  1. Per-token ESM-1v embeddings (final layer 33) of mutated and wild-type sequences.
    Proteins > 1,022 aa are trimmed to a 1,022-aa window centred on the mutation.
  2. SIFT scores (optional).
  3. Features: full-mean, local-mean (13 aa) or mut-site pooling; optional delta (mutated - wild-type) and optional SIFT score.
  4. Grid search with StandardScaler and 5-fold CV grouped by protein family on train.
  5. Evaluation of the best model on the test set.

REQUIREMENTS
  Python 3.11, PyTorch 2.1.2 (CUDA 12.1; GPU recommended)
  pip install -r requirements.txt

  ESM-1v weights (not included):
    wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt

  SIFT (only if USE_SIFT = True; not included):
    sift4g <version> (https://github.com/rvaser/sift4g) and protein database (UniRef90, Viridiplantae)

INPUT
  Output of homology_controlled_splits.py: split_1 ... split_5/{train,test}.tsv and {train,test}_inputs/
  (sequences.fasta, subst_files/; extract subst_files.tar.gz first)

USAGE
  1. Edit the paths and options at the top of main.py
     (SPLITS_DIR, ESM_MODEL, EMB_DIR, SIFT_BIN, SIFT_DB, SIFT_DIR, OUTDIR, EMBEDDING_TYPE, DELTA, USE_SIFT).
  2. Run from this folder:
       python main.py

  Each run is one configuration. To reproduce all results run it 12 times:
  EMBEDDING_TYPE (full-mean | local-mean | mut-site) x DELTA (False | True) x USE_SIFT (False | True).
  Existing embeddings and SIFT predictions are not recomputed.

OUTPUT
  EMB_DIR/                  mut/ and wt/ embeddings (.pt), trim_info.tsv
  SIFT_DIR/split_<n>/{train,test}/  <seq_id>.SIFTprediction
  OUTDIR/<config>/split_<n>/
    train/, test/           features (.pkl), data (.tsv) and CodeCarbon emissions (.csv)
    models/                 best model (.pkl), grid search and CV metrics,
                            train_result.json (test metrics), test_predictions.tsv
