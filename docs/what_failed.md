# What failed

Running log of things that went wrong, feeding the README.

- Phase 0: `meta_*.jsonl` `images` field is a list of dicts, not a dict as first assumed; the
  first parsing attempt crashed. Fixed in `cartrank.data.reviews._first_image`.
