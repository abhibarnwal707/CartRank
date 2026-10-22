# Decisions

## Recommender / demo category: Office_Products (2026-10-22)

Chosen by measured size on the official Amazon Reviews 2023 5-core `last_out` files
(counts in `results/data_profile.md`):

| category | items | interactions | fits target (<=150K items, <=2M inter.) |
|---|---|---|---|
| Office_Products | 77,551 | 1,800,878 | yes |
| Sports_and_Outdoors | 156,235 | 3,472,020 | no |
| Toys_and_Games | 162,035 | 3,861,886 | no |

Electronics and Home_and_Kitchen are far larger and were not counted in full.

Other reasons: it is a mainstream shopping category that ESCI queries cover well, which helps
the ESCI-to-catalog transfer check in Phase 3. 66.8% of items have a price, giving a priced
demo catalog of 51,821 items.

## Recommender splits

The official benchmark ships 5-core `last_out` CSVs. We still build the split in code
(`cartrank.data.splits.leave_last_out`, unit-tested) so the rule is explicit, and it agrees
with the official valid/test files on 99.999% of rows (differences are timestamp ties).

## ESCI splits

US locale and `small_version == 1`. Dev = 10% of train `query_id`s (seed 0). The provided
test split is not used until Phase 7.

## Python

Python 3.12 pinned in `pyproject.toml` (per the playbook), managed with uv.
