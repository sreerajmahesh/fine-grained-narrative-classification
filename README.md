# NaturalLanguageProcessing-Projects

Collection of NLP projects — from subword tokenization and language modeling to media bias analysis.

---

## Projects

| Project | Description | Approach |
|---------|-------------|----------|
| [`wordpiece-fasttext-nplm`](./wordpiece-fasttext-nplm) | WordPiece Tokenizer + FastText + Neural Probabilistic Language Model | Custom WordPiece, char n-gram FastText, feed-forward NPLM |
| [`fine-grained-narrative-classification`](./fine-grained-narrative-classification) | **Fine-Grained Narrative Classification in Biased News** (INDI-PROP, CAA & Farmers' Protest) | DeBERTa-v3-large (description-aware + linear fusion, focal hierarchy loss) + Llama-3.3 reranking — **+0.244 macro-F1 on CAA** vs paper baseline |

See each folder's `README.md` for setup, training, and results.

---

## Highlights — Latest: Fine-Grained Narrative Classification

- **Dataset:** INDI-PROP (1,266 articles, 20 narratives: 11 CAA `C1–C11`, 9 Farmers `F1–F9`, hierarchical bias → narrative)
- **Model:** `microsoft/deberta-v3-large` + dual heads (semantic description head + linear) fused via learnable gate
- **Training:** Stratified split, rare-class augmentation, focal hierarchy loss, per-label thresholds, differential LR + cosine, early stopping
- **Reranking:** High-recall candidate pool → Llama-3.3-70B (Groq) audit → safe fusion (preserve `p≥0.75`)
- **Results:** CAA macro 0.513 (Δ+0.244), Farmers macro 0.221 (= baseline, zero-resource classes)

→ Full details: [`fine-grained-narrative-classification/README.md`](./fine-grained-narrative-classification/README.md) and `docs/` PDFs.

---

## Suggested Repository Structure (if standalone)

If you extract any project as a standalone repo, recommended names:

- **This project:** `fine-grained-narrative-classification` (descriptive, matches paper title) — alternatives: `indi-prop-fanta`, `bias-aware-narrative-classification`
- **Collection:** `NaturalLanguageProcessing-Projects` (this repo)

---

## Setup (general)

```bash
git clone https://github.com/sreerajmahesh/NaturalLanguageProcessing-Projects.git
cd NaturalLanguageProcessing-Projects/<project-folder>
pip install -r requirements.txt
```

Each project has its own `requirements.txt` and instructions.

## License

MIT — datasets remain under original terms (research only).
