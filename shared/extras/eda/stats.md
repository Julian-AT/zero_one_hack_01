# EDA summary

## Sequence lengths

| family | n | mean | std | min | p25 | p50 | p75 | max |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 1000 | 125.3 | 2.5 | 117 | 124 | 125 | 127 | 134 |
| IGBT | 1000 | 148.0 | 2.9 | 139 | 146 | 148 | 150 | 155 |
| IC | 1000 | 115.1 | 2.5 | 107 | 113 | 115 | 117 | 122 |

## Vocabulary

- Total unique steps across all families: **198**
- MOSFET: 137 unique step strings
- IGBT: 147 unique step strings
- IC: 130 unique step strings
- Shared across all 3 families: **94** step strings
- MOSFET-only steps (20): ['ANISOTROPIC ETCH SPACER', 'DEPOSIT SPACER DIELECTRIC', 'EPITAXIAL DEPOSITION', 'EPITAXY ANNEAL', 'EPITAXY PREP', 'GATE OXIDE GROWTH', 'GATE OXIDE PREP', 'IMPLANT LDD']…
- IGBT-only steps (27): ['ALIGN MASK LEVEL 5', 'ALIGN MASK LEVEL 6', 'ANNEAL DIELECTRIC', 'BREAKDOWN VOLTAGE TEST', 'CLEAN AFTER FIELD ETCH', 'CLEAN AFTER OXIDE ETCH', 'CLEAN AFTER WINDOW ETCH', 'DEPOSIT FIELD OXIDE']…
- IC-only steps (29): ['ANNEAL OXIDE', 'ANNEAL POLYSILICON', 'BACKSIDE CLEAN FINAL', 'BACKSIDE THINNING CHECK', 'DEPOSIT BACKSIDE PROTECTION', 'DEPOSIT PAD OXIDE', 'DEPOSIT TUNGSTEN SEED', 'DRY WAFER BACKSIDE']…

## Predictability

**Trigram-with-backoff next-step prediction (no learning, no GPU):**
- Top-1: **0.722**  |  Top-3: **0.968**  |  Top-5: **0.993**  (n=382,294 predictions)

**Position-conditional entropy:**
- MOSFET: mean H = 2.92 bits, max H = 4.10, fraction of positions with H<0.1 (essentially deterministic): 2.2%
- IGBT: mean H = 3.06 bits, max H = 4.23, fraction of positions with H<0.1 (essentially deterministic): 1.9%
- IC: mean H = 2.79 bits, max H = 3.95, fraction of positions with H<0.1 (essentially deterministic): 2.5%

## Cross-family bigram coverage (OOD transfer proxy)

| held-out → | fraction of bigrams seen in other two families |
|---|--:|
| MOSFET | 0.785 |
| IGBT | 0.726 |
| IC | 0.679 |

## Duplicate / 5-gram stats

| family | n_seqs | exact dup | unique 5-grams | 5-grams/seq |
|---|--:|--:|--:|--:|
| MOSFET | 1000 | 0 | 1121 | 121.3 |
| IGBT | 1000 | 0 | 1135 | 144.0 |
| IC | 1000 | 0 | 1539 | 111.1 |

## Plots

- `01_length_distribution.png`
- `02_vocab_overlap.png`
- `03_top30_step_frequency.png`
- `04_category_over_position.png`
- `05_position_entropy.png`
- `06_bigram_coverage.png`
- `07_transition_heatmap.png`