# Fine-Grained Narrative Classification in Biased News

> **DeBERTa + Description-Aware Fusion + LLM Reranking for INDI-PROP**

This project addresses **fine-grained narrative classification** on Indian news media — identifying the underlying storyline that frames an article's ideological message. It is built on the **INDI-PROP** dataset (1,266 articles, CAA & Farmers' Protest) and implements a **single-stage multi-label classifier** with hierarchy-aware training and Llama-3.3-based reranking, outperforming the published DeBERTa baseline on CAA (+0.244 macro-F1).

- **Paper (course):** `docs/finegrainclassification.pdf` — *Fine-grained Narrative Classification in Biased News Articles* (FANTA, TPTC, INDI-PROP)
- **Project report:** `docs/NLP_FINAL_PROJECT.pdf` — full problem, methodology, results for this implementation

---

## Problem Statement

Digital news has amplified biased and propagandistic framing. Shallow signals (sentiment, isolated techniques) miss the **deeper narrative structure** that legitimizes one view while delegitimizing the other.

**Tasks:**
1. **Article Bias Classification** (hierarchical pre-step): `Pro-Government` / `Pro-Opposition` / `Neutral`
2. **Fine-Grained Narrative Classification** (this project): For biased articles, assign one or more **event-specific narrative labels** (11 for CAA: `C1–C11`, 9 for Farmers: `F1–F9`). Neutral articles have `No Narrative`.
3. **Persuasive Technique Classification** (20 techniques, see paper)

Challenges: implicit/contextual narratives, dependence on bias, extreme class imbalance (<10 examples for some classes), neutral → no narrative.

## Dataset: INDI-PROP

Modified from INDI-PROP for Indian news (CAA 2019–2024, Farmers' Protest 2020–2024). Sources: OpIndia, Republic World, Swarajya, The Quint, Hindustan Times (MBFC spectrum). Split by the authors.

| Field | Description |
|-------|-------------|
| `text` | Full article body |
| `Event` | `CAA` or `Farmers_Protests` |
| `Bias` | `Pro government` / `Pro opposition` / `Neutral` (dropped for this project’s single-stage model) |
| `Narrative` | Comma/ID-separated narrative labels (e.g., `C1, C5` or `Glorification of CAA`) or `No narrative` |
| `Persuasive techniques` | Span-level technique labels (not used here) |

**Label taxonomies** (`src/labels.py`):

- **CAA (11):** `C1` Glorification of central govt, `C2` Vilification of opposition, `C3` Glorification of CAA, `C4` Delegitimization of Critics, `C5` Framing anti-CAA protests as Subversive, `C6` Opposition spreading misinformation, `C7` Pre-planned conspiracy, `C8` Vilification of central govt (pro-opp), `C9` Vilification of CAA, `C10` Glorifying anti-CAA protesters, `C11` Framing protesters as victims. `side`: `pro_govt` (C1–C7) vs `pro_opp` (C8–C11).
- **Farmers (9):** `F1` Glorification of central govt, `F2` Vilification of opposition, `F3` Justifying farm laws by critiquing old policies, `F4` Criticizing global figures, `F5` Vilification of central govt (pro-opp), `F6` Depicting farmers as victims, `F7` Framing protests as subversive (counter), `F8` Accusing media/govt of manipulation, `F9` Emphasizing global endorsements.

See `docs/finegrainclassification.pdf` Table 2 for train/test counts and Figure 2 for taxonomy trees.

---

## Methodology

### 1. Baselines (mid-project)
Fine-tuned **BERT** (`bert-base-uncased`) and **RoBERTa** (`roberta-base`) with unified pipeline (lr 2e-5, bs 8, 5 epochs, wd 0.01). Single-label (bias) vs multi-label (narrative/technique, sigmoid thr 0.2).

### 2. Proposed Method — This Implementation

**Problem as multi-label:** Input article → one or more narrative labels.

**Model (`src/model.py` — `NarrativeModel`):**
- Backbone: **`microsoft/deberta-v3-large`** (hidden 1024)
- Mean pooling over last hidden state
- Two heads:
  - **(a) Description-based head** (`LabelDescHead`): Each label’s `name + description` is encoded via the same DeBERTa encoder (frozen init from tokenizer), L2-normalized, cosine similarity `article_emb @ label_emb / tau + bias`. Helps rare labels via semantic prior. `tau = exp(log_tau)` ∈ [1e-3, 5], learnable.
  - **(b) Linear head:** Standard `Linear(hidden, K)` — strong for frequent classes.
  - **(c) Fusion:** `logit = sigmoid(gate) * logit_desc + (1 - sigmoid(gate)) * logit_lin`, `gate` is `K`-dim learnable (initialized 0 → 0.5 weight).
- Dropout 0.2

**Training (`src/train.py`):**
- Loss: **FocalHierarchyLoss** — BCE with `pos_weight` (neg/pos clipped [1,10]), focal `alpha=0.5, gamma=1`, hierarchy mask `M[i,j]=1 if side_i==side_j` with `cross_side_penalty` (weight wrong-side negatives higher when has_gold), `class_active_mask` (ignore labels with 0 train examples).
- Data: `load_splits` → `binarize` → `make_val_split` (MultilabelStratifiedShuffleSplit 85/15, fallback random) → `augment_rare` (for classes <25, duplicate with sentence/word dropout).
- Optim: **AdamW** with differential LR — encoder `lr=5e-6`, heads `lr*20`, wd 0.01 / 0.0, cosine schedule 15% warmup, `GradScaler` + AMP, `accum=4`, `clip 1.0`, early stopping patience 6 on val macro-F1.
- Thresholding: `tune_thresholds` grid `0.05:0.025:0.96` per-class maximizing F1 on val; inactive/0-support → `0.95`.

**LLM Reranking — post-processing (`src/rerank.py`):**
- `Model → High Recall → LLM → Precision Refinement`
- Candidate pool: `prob >= max(thr - 0.20, 0.05)` (looser), `HIGH_CONF=0.75` preserved always
- LLM (`llama-3.3-70b-versatile` via Groq): prompt with article (3000 chars) + candidates + descriptions + confidences, JSON `{"keep": [...]}`. Prompt encourages keeping unless clearly incorrect. Safe fusion: high-conf from model always kept.

### Original Paper’s FANTA (reference)
Multi-hop pipeline: NER + coreference + relation extraction → context framing → bias (3-way) → narrative (event-specific taxonomy). TPTC for techniques (coarse G1–G7 → fine-grained). Not reimplemented here; this project uses the single-stage DeBERTa fusion instead.

---

## Results

**Test (our reranked model vs paper’s DeBERTa baseline, no oracle bias):**

| Event | Micro | Macro | Weighted | Oracle | Reranker |
|-------|-------|-------|----------|--------|----------|
| CAA – DeBERTa (paper) | 0.538 | 0.269 | 0.471 | no | None |
| **CAA – Ours** | **0.546** | **0.513** | **0.653** | no | Llama-3.3-70B | **+0.244 macro** |
| CAA – FANTA-GPT-4o | 0.670 | 0.660 | 0.700 | yes | – |
| Farmers – DeBERTa (paper) | 0.669 | 0.221 | 0.637 | no | None |
| **Farmers – Ours** | 0.260 | 0.221 | 0.339 | no | Llama-3.3-70B | = baseline under harder setting |
| Farmers – FANTA-GPT-4o | 0.730 | 0.700 | 0.780 | yes | – |

- CAA: large gain from focal loss + per-label thr + reranking.
- Farmers: matches baseline despite no oracle, extreme sparsity (F4, F8, F9: 17 train examples total → F1=0), weaker reranker (Llama-3.3 via Groq vs GPT-4o-mini). Excluding zero-resource classes, avg per-class F1 ≈ 0.315.

See `docs/NLP_FINAL_PROJECT.pdf` § Results for per-class table.

---

## Project Structure

```
fine-grained-narrative-classification/
├── src/
│   ├── labels.py   # CAA_LABELS, FARMERS_LABELS, get_labels()
│   ├── data.py     # load_raw, binarize, augment_rare, make_val_split, load_splits
│   ├── model.py    # BACKBONE=deberta-v3-large, LabelDescHead, NarrativeModel, FocalHierarchyLoss
│   ├── train.py    # TextDS, tune_thresholds, eval_metrics, run(event, train_path, test_path)
│   └── rerank.py   # Groq Llama reranker, build_prompt, llm_audit, rerank(event)
├── docs/
│   ├── NLP_FINAL_PROJECT.pdf
│   └── finegrainclassification.pdf
├── data/           # put train.csv / test.csv here (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

**File renames:** Original `data-2.py` → `data.py`, `labels-2.py` → `labels.py`, `model-3.py` → `model.py`, `train-3.py` → `train.py`; `rerank.py` sanitized (no hardcoded keys).

---

## Setup & Usage

### Prerequisites
- Python 3.10+, CUDA GPU recommended (DeBERTa-large)
- Groq API key for reranking (optional) — https://console.groq.com

### Install

```bash
git clone https://github.com/sreerajmahesh/NaturalLanguageProcessing-Projects.git
cd NaturalLanguageProcessing-Projects/fine-grained-narrative-classification

pip install -r requirements.txt
# or
pip install torch transformers datasets scikit-learn pandas numpy iterative-stratification groq
```

### Data

Prepare CSVs with at least columns `text`/`Article` and `narrative`/`Narrative` (bias column is dropped). Example:

```
text,narrative
"The government stands firm...",C1; C3
"Protesters face jail...",C11
"Elderly farmers spend nights...",F6
```

Place as `data/train_caa.csv`, `data/test_caa.csv`, etc., or pass explicit paths.

### Train

```bash
# CAA event
python src/train.py --event caa --train_path data/train_caa.csv --test_path data/test_caa.csv --epochs 20 --bs 4 --lr 5e-6 --max_len 512

# Farmers
python src/train.py --event farmers --train_path data/train_farmers.csv --test_path data/test_farmers.csv

# Outputs (gitignored): best_<event>.pt, test_logits_<event>.npy, test_y_<event>.npy, test_texts_<event>.json, results_<event>.json
```

### Rerank (optional LLM refinement)

```bash
export GROQ_API_KEY="gsk_..."
# or for rotation: export GROQ_API_KEYS="gsk_...,gsk_...,gsk_..."

python src/rerank.py caa        # reads best_caa.pt etc., writes results_rerank_caa.json
python src/rerank.py farmers
```

If no Groq key is set, `src/rerank.py` will warn and skip.

### Baselines (BERT/RoBERTa)

Not included in this folder — see `docs/NLP_FINAL_PROJECT.pdf` §1 for hyperparameters (bert-base-uncased / roberta-base, lr 2e-5, bs 8, 5 epochs).

---

## Key Files Explained

| File | Purpose |
|------|---------|
| `src/labels.py` | 11 CAA + 9 Farmers label dicts with `id`, `side`, `name`, `description` |
| `src/data.py:binarize` | Finds text/label cols, parses `C\d+`/`F\d+` or name matching, returns `texts, Y` |
| `src/data.py:augment_rare` | For <25 examples, duplicate with sentence dropout (keep first/last) or word dropout |
| `src/model.py:LabelDescHead` | Encodes label descriptions, cosine sim / tau |
| `src/model.py:FocalHierarchyLoss` | BCE + focal + hierarchy wrong-side penalty |
| `src/train.py:tune_thresholds` | Per-class grid search 0.05–0.95 step 0.025 |
| `src/rerank.py` | Llama prompt + candidate filtering, safe fusion (`HIGH_CONF=0.75`) |

---

## Ethics & Limitations

- Articles from public news; no PII beyond published content. Dataset intentionally includes biased/offensive framing — for research only.
- Labels are subjective; inter-rater κ bias 0.611, narrative mean 0.605–0.613. Treat as fallible.
- Do not use for censorship/political profiling.
- Limited to 2 events, English only, text-only (no images), DeBERTa-large compute heavy.

---

## Citation

If using INDI-PROP/FANTA, cite the paper in `docs/finegrainclassification.pdf` (arXiv:2512.03582v1):

```
@inproceedings{afroz2025finegrained,
  title={Fine-grained Narrative Classification in Biased News Articles},
  author={Afroz et al.},
  year={2025}
}
```

## Suggested Standalone Repo Name

If publishing this project separately, recommended name: **`fine-grained-narrative-classification`** (descriptive, matches paper). Alternatives: `indi-prop-fanta` (short, references dataset + FANTA), `bias-aware-narrative-classification`.

## License

MIT — dataset remains under its original release terms (research only). Model code MIT.

## Credits

- Authors: Afroz, Vardhan, Bhakuni, Punia, Kumar, Akhtar (IIITD + BEL)
- This implementation: stratified split, DeBERTa-v3-large, fusion gate, focal hierarchy, per-label thr, Groq Llama rerank.
