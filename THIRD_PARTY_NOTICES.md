# Third-party notices

Licence terms below were read from each source's own page on 2026-10-24. Where a page states no
licence, that is said plainly; it is not a grant of rights. Check the current terms before relying
on this.

## Datasets

**Amazon Reviews 2023** (McAuley Lab, UCSD): https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
The dataset card states **no licence** and has no terms-of-use section (only Hugging Face's generic
site terms are linked). It asks for citation of Hou et al., 2024, *Bridging Language and Items for
Retrieval and Recommendation* (arXiv:2403.03952). CartRank uses the Office_Products category for
the recommender and the demo catalog (titles, brands, prices, ratings, feature text, image URLs).
Because no licence is stated, this project treats the data as research use only and does not
redistribute the raw dataset. See `docs/decisions.md` for how derived catalog files are shared.

**Amazon Shopping Queries Dataset (ESCI)**: https://github.com/amazon-science/esci-data
The repository states "This project is licensed under the Apache-2.0 License." The README gives no
separate licence for the dataset files. Citation requested: Reddy et al., 2022, *Shopping Queries
Dataset: A Large-Scale ESCI Benchmark for Improving Product Search* (arXiv:2206.06588). Used to
train and evaluate search; raw files are not redistributed.

## Models

| Model | Use | Licence (as shown on its model page) |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | base bi-encoder; the fine-tuned CartRank encoder derives from it | MIT |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | reranker (used as published, not fine-tuned) | Apache-2.0 |

## Services

- Groq API (`openai/gpt-oss-20b`) and OpenRouter free models are used through their public APIs
  under each provider's terms. API keys are kept out of the repository.

## Software

Main libraries: sentence-transformers, PyTorch, bm25s, FAISS, pandas, pydantic, Streamlit, httpx.
Each carries its own open-source licence; see the respective projects.

## Images and video in the app

- `app/static/img/*.jpg`: nine photographs from the WordPress Photo Directory (wordpress.org/photos), released
  under CC0. Sources are listed in `app/static/img/CREDITS.md`.
- `app/static/video/pencil-sharpener.webm` (and its poster frame): "Aerra's Films of the Ordinary I --
  Staedtler Pencil Sharpener" by Aerra Carnicom, CC BY-SA 4.0, via Wikimedia Commons
  (https://commons.wikimedia.org/wiki/File:Aerra%27s_Films_of_the_Ordinary_I_--_Staedtler_Pencil_Sharpener.webm).
  Shown unaltered; credited in the app footer. Record: `app/static/video/credits.json`.
- Product photos are not stored in the repo; the app shows them from the Amazon listing URLs in the catalog.
