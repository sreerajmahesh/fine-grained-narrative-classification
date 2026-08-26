import json
import os
import numpy as np
from groq import Groq
from labels import get_labels


MODEL = "llama-3.3-70b-versatile"

# Load Groq API keys from environment
# Set GROQ_API_KEY for single key, or GROQ_API_KEYS (comma-separated) for multiple keys for rotation
# Example: export GROQ_API_KEY="gsk_..."
# Or for rotation: export GROQ_API_KEYS="gsk_...,gsk_...,gsk_..."
def _load_clients():
    keys = []
    multi = os.getenv("GROQ_API_KEYS", "")
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip().startswith("gsk_")]
    single = os.getenv("GROQ_API_KEY", "")
    if single and single not in keys:
        keys.append(single.strip())
    if not keys:
        # Placeholder - will fail gracefully at runtime if not set
        # Do NOT hardcode keys here. Set env var.
        print("[warn] GROQ_API_KEY(S) not set. Set GROQ_API_KEY or GROQ_API_KEYS env var.")
        return []
    return [Groq(api_key=k) for k in keys]

clients = _load_clients()
_idx = [0]

def get_client():
    if not clients:
        raise RuntimeError("No Groq clients configured. Set GROQ_API_KEY env var.")
    c = clients[_idx[0]]
    _idx[0] = (_idx[0] + 1) % len(clients)
    return c

def build_prompt(article, candidates, probs, label_meta):
    lines = []
    lines.append("You are a narrative classification auditor.")
    lines.append("A model predicted the following narratives with confidence scores.")
    lines.append("Your job: REMOVE only clearly incorrect labels.")
    lines.append("IMPORTANT: If unsure, KEEP the label.")
    lines.append("Do NOT remove labels just because evidence is weak.")
    lines.append("")

    lines.append("=== Narratives ===")
    for lid, p in zip(candidates, probs):
        meta = next(l for l in label_meta if l["id"] == lid)
        lines.append(f"{lid} (confidence={p:.2f}): {meta['name']} — {meta['description']}")

    lines.append("")
    lines.append("=== Article ===")
    lines.append(article[:3000])
    lines.append("")

    lines.append("Return ONLY JSON:")
    lines.append('{"keep": ["label_ids"]}')

    return "\n".join(lines)

def llm_audit(article, candidates, probs, label_meta, retries=3):
    if not candidates:
        return []

    prompt = build_prompt(article, candidates, probs, label_meta)

    for _ in range(retries):
        try:
            client = get_client()
            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            obj = json.loads(res.choices[0].message.content)
            keep = obj.get("keep", [])

            return [k for k in keep if k in candidates]

        except Exception:
            continue

    return candidates


def rerank(event):
    label_meta = get_labels(event)
    label_ids = [l["id"] for l in label_meta]

    logits = np.load(f"test_logits_{event}.npy")
    y_true = np.load(f"test_y_{event}.npy")
    texts = json.load(open(f"test_texts_{event}.json"))

    ckpt = __import__("torch").load(f"best_{event}.pt", map_location="cpu")
    thr = np.array(ckpt["thresholds"])

    probs = 1 / (1 + np.exp(-logits))

    base_pred = (probs >= thr).astype(int)

    loose_thr = np.maximum(thr - 0.20, 0.05)
    candidate_pool = (probs >= loose_thr).astype(int)

    final_pred = base_pred.copy()

    HIGH_CONF = 0.75

    for i in range(len(texts)):
        cand_idx = np.where(candidate_pool[i] == 1)[0]
        if len(cand_idx) == 0:
            continue

        cand_labels = [label_ids[j] for j in cand_idx]
        cand_probs = [probs[i, j] for j in cand_idx]

        kept = llm_audit(texts[i], cand_labels, cand_probs, label_meta)

        new_row = np.zeros(len(label_ids))

        for j, lid in enumerate(label_ids):
            prob = probs[i, j]

            if prob >= HIGH_CONF:
                new_row[j] = 1
                continue

            if lid in kept:
                new_row[j] = 1

        final_pred[i] = new_row

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(texts)}] cand={len(cand_labels)} kept={len(kept)}")

    from sklearn.metrics import precision_recall_fscore_support

    res = {}
    for avg in ["micro", "macro", "weighted"]:
        _, _, f, _ = precision_recall_fscore_support(
            y_true, final_pred, average=avg, zero_division=0
        )
        res[f"{avg}_f1"] = float(f)

    _, _, per_class, _ = precision_recall_fscore_support(
        y_true, final_pred, average=None, zero_division=0
    )

    res["per_class_f1"] = {
        label_ids[i]: float(per_class[i]) for i in range(len(label_ids))
    }

    print(f"\n[RERANK {event}] micro={res['micro_f1']:.3f} macro={res['macro_f1']:.3f} weighted={res['weighted_f1']:.3f}")
    for lid, f in res["per_class_f1"].items():
        print(f"  {lid}: {f:.3f}")

    with open(f"results_rerank_{event}.json", "w") as f:
        json.dump(res, f, indent=2)

    return res


if __name__ == "__main__":
    import sys
    rerank(sys.argv[1])
