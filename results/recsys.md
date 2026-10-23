# Recommender results (Office_Products, dev split)

Temporal leave-last-out per user: 223,308 users, 77,551 items, 1,354,262 train interactions. Metrics are **full ranking over all 77,551 items** (no sampled negatives), the user's train items excluded, ties counted against the model. Test split not used (Phase 7).

| Model | HR@10 | NDCG@10 | Δ NDCG vs previous row | Cold HR@10 | Cold NDCG@10 | Train s | Setting |
|---|---|---|---|---|---|---|---|
| Popularity | 0.0111 | 0.0062 | - | 0.0000 | 0.0000 | 8 | train counts |
| ItemKNN | 0.0269 | 0.0178 | +0.0116 | 0.0109 | 0.0065 | 8 | cosine, top-200 neighbours |
| BPR-MF (PyTorch) | 0.0260 | 0.0147 | -0.0031 | 0.0002 | 0.0001 | 543 | dim 128, lr 0.003, reg 0.001, epoch 66 |
| Two-tower, ID-only item tower | 0.0392 | 0.0228 | +0.0080 | 0.0003 | 0.0002 | 45 | tau 0.1, dim 128, epoch 6 (Kaggle GPU) |
| Two-tower, ID + text item tower | 0.0462 | 0.0267 | +0.0039 | 0.0005 | 0.0002 | 64 | tau 0.1, dim 128, epoch 7 (Kaggle GPU) |

Cold slice: the 22,514 dev users whose target item has fewer than 5 train interactions (about 10% of users). Train seconds: popularity and ItemKNN on the local CPU, BPR on the local MPS GPU for its selected configuration, two-tower on a Kaggle GPU summed over all epochs run (before early stopping), so times are not comparable across hardware.

## Ablation: item tower ID-only vs ID + text embedding

| Item tower | Best tau | HR@10 | NDCG@10 | Cold HR@10 | Cold NDCG@10 |
|---|---|---|---|---|---|
| ID-only | 0.1 | 0.0392 | 0.0228 | 0.0003 | 0.0002 |
| ID + text | 0.1 | 0.0462 | 0.0267 | 0.0005 | 0.0002 |

Adding the text embedding changes overall NDCG@10 by +0.0039 and HR@10 by +0.0071. On the cold slice both two-tower variants are near zero in absolute terms (HR@10 0.0003 and 0.0005), far below ItemKNN (0.0109); the text signal helps overall ranking but does not rescue cold items in this setup. Likely cause (not verified): cold items keep an untrained random ID embedding that adds noise to their vector.

## Two-tower runs (dev, best epoch chosen by dev NDCG@10, patience 3)

| Run | Best epoch | Epochs run | HR@10 | NDCG@10 |
|---|---|---|---|---|
| id_only_tau0.05 | 4 | 7 | 0.0352 | 0.0205 |
| id_only_tau0.1 | 6 | 9 | 0.0392 | 0.0228 |
| id_text_tau0.05 | 4 | 7 | 0.0436 | 0.0253 |
| id_text_tau0.1 | 7 | 10 | 0.0462 | 0.0267 |

The best temperature (0.1) is the largest tried, so the optimum may lie beyond the grid.

## ItemKNN tuning (dev)

| k neighbours | HR@10 | NDCG@10 |
|---|---|---|
| 50 | 0.0260 | 0.0173 |
| 100 | 0.0265 | 0.0176 |
| 200 | 0.0269 | 0.0178 |

The best k is the largest tried (200), so the optimum may lie beyond the grid.

## BPR tuning (20,000-user dev subset, NDCG@10)

| dim | lr | reg | best epoch | subset NDCG@10 | seconds |
|---|---|---|---|---|---|
| 64 | 0.003 | 0.0001 | 28 | 0.0050 | 136 |
| 64 | 0.003 | 0.001 | 60 | 0.0089 | 507 |
| 64 | 0.003 | 0.01 | 16 | 0.0082 | 87 |
| 64 | 0.01 | 0.0001 | 6 | 0.0045 | 47 |
| 64 | 0.01 | 0.001 | 44 | 0.0067 | 202 |
| 64 | 0.01 | 0.01 | 8 | 0.0070 | 57 |
| 128 | 0.003 | 0.0001 | 60 | 0.0079 | 436 |
| 128 | 0.003 | 0.001 | 60 | 0.0134 | 441 |
| 128 | 0.003 | 0.01 | 16 | 0.0102 | 166 |
| 128 | 0.01 | 0.0001 | 16 | 0.0054 | 165 |
| 128 | 0.01 | 0.001 | 34 | 0.0095 | 303 |
| 128 | 0.01 | 0.01 | 6 | 0.0082 | 91 |
