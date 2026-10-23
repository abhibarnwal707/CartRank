# Kaggle fine-tune: bge-small on ESCI

Notebook: `notebooks/kaggle/finetune_bge_esci.ipynb`. It fine-tunes `BAAI/bge-small-en-v1.5` as a
bi-encoder with `MultipleNegativesRankingLoss` on ESCI (query, Exact product) pairs.

## Run it

1. Kaggle, **Create > New Notebook > File > Import Notebook**, upload the `.ipynb`.
2. Right sidebar **Session options**:
   - **Accelerator: GPU T4 x2** (or P100). The notebook uses one GPU, fp16.
   - **Internet: On** (needs a phone-verified Kaggle account). The notebook downloads ESCI
     (about 1.2 GB) from GitHub and the model from Hugging Face; both are public, so no token is
     needed and no secrets should be added.
   - **Persistence**: "Variables and Files" is optional; checkpoints are written to
     `/kaggle/working` and the trainer resumes from the latest one if you re-run the training cell
     after a restart.
3. **Run All**. Check these as it goes:
   - The split cell must print `train queries 18799 dev queries 2089`. It asserts a checksum of the
     dev query ids against the local repo, so a mismatch fails loudly instead of leaking dev data.
   - The text cell must print `482105 products`.
   - The trainer prints a loss every 50 steps.
4. Outputs (right sidebar, **Output**, or the `/kaggle/working` panel):
   - `bge-small-esci-ft.zip` (the model, a few tens of MB)
   - `dev_check.json` (quick base vs fine-tuned NDCG@10 on dev, rerank mode)
   - `train_log.json` (loss history)

## Settings you may change (config cell)

| Setting | Default | Notes |
|---|---|---|
| `EPOCHS` | 1 | Playbook says 1-2. Change `RUN_NAME` when you run a second configuration. |
| `BATCH_SIZE` | 128 | Larger gives more in-batch negatives; lower it if you get an out-of-memory error. |
| `LR` | 5e-5 | |
| `USE_HARD_NEG` | False | True trains on (query, Exact, Irrelevant) triplets. Only 10,657 of 18,799 train queries have an Irrelevant product, so this uses fewer examples. |

Both modes were smoke-tested locally on 512 examples (sentence-transformers 6.2.0). The runtime and
final quality on Kaggle are not measured yet; do not quote any number until the dev evaluation below
has run.

## Bring the model back

1. Download `bge-small-esci-ft.zip` (and `dev_check.json`) from the notebook's Output tab.
2. Unzip into `CartRank/artifacts/models/bge-small-esci-ft/` (git-ignored). That folder should
   contain `config.json`, `model.safetensors`, `modules.json` and so on.
3. Tell Claude. It will embed the 482,105 products with the fine-tuned encoder (about 26 minutes on
   Apple MPS for the base model), re-tune the hybrid, re-run the cross-encoder rerank and add the
   rows to `results/search_baselines.md`.

The model is **not** pushed to the Hugging Face Hub until the owner approves.
