from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROCESSED = DATA / "processed"

ESCI_DIR = DATA / "esci"
REVIEWS_DIR = DATA / "reviews"

# Chosen in Phase 0 by measured size after 5-core filtering; see docs/decisions.md.
CATEGORY = "Office_Products"
