# Cross-encoder batch size vs memory

Resident memory of the serving pipeline on a laptop CPU (LLM off) after loading and after 50 demo queries. The top-10 order was identical between batch 64 and batch 4 on 25 of 25 queries, so the app uses batch 4. Budget: 2,000 MB.

| CE batch | RSS after load MB | RSS after 50 queries MB |
|---|---|---|
| 2 | 1,130 | 1,373 |
| 4 | 1,130 | 1,421 |
| 8 | 1,131 | 1,560 |
| 16 | 1,116 | 1,743 |
| 64 | 1,128 | 2,406 |
