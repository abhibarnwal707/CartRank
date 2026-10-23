# Search error analysis (ESCI dev, full-corpus mode)

Final system = hybrid (BM25 + fine-tuned dense, RRF) + cross-encoder rerank of the top 50. It scores
below BM25 on NDCG@10 for **468 of 2089 dev queries (22.4%)**.
This file covers a random sample of 30 of those 468 (seed 0). Raw per-query data:
`results/search_error_analysis_raw.json`.

**How to read this.** The reasons below are my own manual labels from reading each query's top-5
results for both systems; they are judgements, not measurements. In full-corpus mode a product
without an ESCI judgement counts as non-relevant, so a loss can mean "final system retrieved
plausible but unlabelled products". Across the sample the top 10 contains on average 5.0
unjudged products for BM25 and 5.6 for the final system.

## Summary

| Reason group | Queries |
|---|---|
| Near-tie reorder: NDCG gap <= 0.07, no clear error | 16 |
| Top results mostly unjudged (labelling gap) | 3 |
| Specific failure modes (negation, dropped attribute, numeric specs, rare token, ambiguity) | 11 |

Specific failure modes seen: negation or exclusion phrases ("without drilling", "without cargo",
"without overflow") handled lexically, so the model matches the phrase rather than the constraint;
an attribute such as colour dropped ("mint green"); numeric specs and fitment ("4mm 6mm 8mm",
"04 ford ranger"); rare exact tokens that BM25 matches and the dense model blurs ("bricktober");
ambiguous or very short queries; and long attribute-stacked queries. These are the cases the
Phase 5 query parser (hard filters for price, brand, must-have, exclude) is designed to help.

## Sample

| # | Query | NDCG BM25 | NDCG final | Gap | Unjudged top-10 (BM25 / final) | Reason |
|---|---|---|---|---|---|---|
| 0 | plain canvas tote bag for kids | 0.286 | 0.162 | 0.124 | 6 / 5 | top results unjudged (plausible but unlabelled) |
| 1 | saints football merchandise | 0.269 | 0.110 | 0.159 | 6 / 9 | top results unjudged; vague query ('merchandise') |
| 2 | mindspace office desk organizer | 0.934 | 0.870 | 0.064 | 1 / 2 | near-tie reorder (gap <= 0.07), no clear error |
| 3 | leica m monochrome | 0.999 | 0.961 | 0.038 | 1 / 1 | near-tie reorder (gap <= 0.07), no clear error |
| 4 | roadtripper | 0.989 | 0.968 | 0.020 | 2 / 2 | near-tie reorder (gap <= 0.07), no clear error |
| 5 | curtain rods without drilling holes | 0.543 | 0.347 | 0.196 | 5 / 5 | negation/constraint ('without drilling'); a Complement ranked first |
| 6 | water turtle | 0.449 | 0.414 | 0.035 | 7 / 8 | near-tie reorder (gap <= 0.07), no clear error |
| 7 | ir air compressor | 0.359 | 0.307 | 0.052 | 8 / 6 | near-tie reorder (gap <= 0.07), no clear error |
| 8 | ted nasmith | 0.586 | 0.531 | 0.055 | 5 / 5 | near-tie reorder (gap <= 0.07), no clear error |
| 9 | flesh lightsex toy male sucking | 0.581 | 0.526 | 0.054 | 7 / 7 | near-tie reorder (gap <= 0.07), no clear error |
| 10 | ninjago bricktober | 0.563 | 0.267 | 0.297 | 2 / 7 | rare exact token diluted ('bricktober'); generic Ninjago sets unjudged |
| 11 | christmas pillow covers | 0.606 | 0.000 | 0.606 | 3 / 10 | top-10 all unjudged (near-duplicate listings), so NDCG 0 is a labelling gap |
| 12 | tweezer coil | 0.942 | 0.928 | 0.014 | 5 / 5 | near-tie reorder (gap <= 0.07), no clear error |
| 13 | 04 ford ranger third brake light without cargo | 0.536 | 0.352 | 0.184 | 1 / 1 | negation + vehicle fitment ('without cargo', '04 ford ranger'); a Substitute ranked first |
| 14 | shoes uppers | 0.146 | 0.000 | 0.146 | 9 / 10 | ambiguous short query ('shoes uppers'); drifted to cycling shoes |
| 15 | large chess board table | 0.406 | 0.340 | 0.066 | 6 / 8 | near-tie reorder (gap <= 0.07), no clear error |
| 16 | 1-1/4 lavatory drain without overflow | 0.319 | 0.095 | 0.223 | 6 / 7 | negation + size spec ('without overflow', '1-1/4'); matched the literal phrase on other products |
| 17 | lego street signs | 0.505 | 0.494 | 0.011 | 4 / 3 | near-tie reorder (gap <= 0.07), no clear error |
| 18 | proclick | 1.000 | 0.901 | 0.099 | 0 / 0 | product-line term ('proclick'); a Substitute ranked above an Exact |
| 19 | cleveland indians onsie | 0.702 | 0.653 | 0.048 | 3 / 2 | near-tie reorder (gap <= 0.07), no clear error |
| 20 | black lace tank for women | 0.523 | 0.476 | 0.047 | 6 / 6 | near-tie reorder (gap <= 0.07), no clear error |
| 21 | mint green rugs for girls rooms | 0.738 | 0.070 | 0.668 | 9 / 8 | attribute dropped (colour 'mint green'); returned a navy rug |
| 22 | 4mm 6mm 8mm tungsten wedding ring imitated | 0.430 | 0.310 | 0.120 | 7 / 7 | numeric spec tokens ('4mm 6mm 8mm'); mostly unjudged top results |
| 23 | mens black under armour socks | 0.711 | 0.653 | 0.057 | 2 / 3 | near-tie reorder (gap <= 0.07), no clear error |
| 24 | maxi skirt bikini cover | 0.460 | 0.323 | 0.137 | 6 / 5 | ambiguous multi-concept query ('maxi skirt bikini cover') |
| 25 | sexy ruffle off shoulder holiday beach mini short dress | 0.322 | 0.139 | 0.182 | 9 / 8 | long attribute-stacked query; drifted to a partial match (holiday print) |
| 26 | &a pedicure spa parts, pop-up drain plug without strainer | 0.066 | 0.006 | 0.060 | 9 / 9 | near-tie reorder (gap <= 0.07), no clear error |
| 27 | ipad 7th generation heavy duty case with screen protector | 0.591 | 0.564 | 0.028 | 4 / 5 | near-tie reorder (gap <= 0.07), no clear error |
| 28 | 100ct wide ruled paper | 0.159 | 0.156 | 0.003 | 8 / 8 | near-tie reorder (gap <= 0.07), no clear error |
| 29 | spy gadgets | 0.735 | 0.630 | 0.105 | 4 / 5 | short generic query ('spy gadgets'); reordered Exact items |

## Notes and limits

- 30 queries is a small sample; group counts are indicative, not rates over all dev queries.
- A keyword heuristic for tags (brand in query, numbers, negation, length) is in
  `eval/error_analysis.py`; the brand tag fires on most queries and was not used for the labels.
- Overall, the final system beats BM25 on dev by +0.0898 NDCG@10 (0.3172 to 0.4070); these losses are
  the minority case.
