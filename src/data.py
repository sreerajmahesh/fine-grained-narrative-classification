import numpy as np
import pandas as pd
import re
import random
from labels import get_labels



def load_raw(event, train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for col in ["Bias", "bias", "Stance", "stance"]:
        if col in train.columns: train = train.drop(columns=[col])
        if col in test.columns: test = test.drop(columns=[col])
    return train, test



def _parse_labels(cell, label_meta):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s.lower() in ["no narrative", "none", "", "nan", "neutral"]:
        return []

    valid_ids = {l["id"] for l in label_meta}
    name_to_id = {l["name"].lower().strip(): l["id"] for l in label_meta}
    name_to_id["framing anti-caa protests as sam"] = "C5"

    
    ids_found = [x for x in re.findall(r"\b[CF]\d+\b", s) if x in valid_ids]
    if ids_found:
        return list(set(ids_found))
    parts = re.split(r"[;,|\n]+", s)
    found = []
    for p in parts:
        p_norm = p.strip().lower().strip(".'\"")
        if p_norm in name_to_id:
            found.append(name_to_id[p_norm])
            continue
        for name, lid in name_to_id.items():
            if p_norm == name:
                found.append(lid); break
            if len(p_norm) > 15 and len(name) > 15:
                if p_norm in name and abs(len(p_norm) - len(name)) < 15:
                    found.append(lid); break
                if name in p_norm and abs(len(p_norm) - len(name)) < 15:
                    found.append(lid); break
    return list(set(found))


def _find_col(df, candidates, kind):
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    raise ValueError(f"Could not find {kind} column. Available: {list(df.columns)}")


def binarize(df, label_meta):
    text_col = _find_col(df, ["text", "Article", "article", "content", "body"], "text")
    label_col = _find_col(df, ["narrative", "Narrative", "label", "labels"], "label")
    print(f"  text_col={text_col}  label_col={label_col}")

    label_ids = [l["id"] for l in label_meta]
    idmap = {lid: i for i, lid in enumerate(label_ids)}
    texts = df[text_col].astype(str).tolist()

    N, K = len(df), len(label_ids)
    Y = np.zeros((N, K), dtype=np.float32)
    for i, cell in enumerate(df[label_col].tolist()):
        for lid in _parse_labels(cell, label_meta):
            if lid in idmap:
                Y[i, idmap[lid]] = 1.0
    return texts, Y


def augment_rare(texts, Y, min_count=25, seed=42):
    rng = random.Random(seed)
    counts = Y.sum(axis=0)
    out_texts, out_Y = list(texts), [y.copy() for y in Y]
    for c in range(Y.shape[1]):
        if counts[c] == 0 or counts[c] >= min_count:
            continue
        pos_idx = np.where(Y[:, c] == 1)[0]
        needed = int(min_count - counts[c])
        for _ in range(needed):
            src = int(rng.choice(pos_idx))
            sents = re.split(r'(?<=[.!?])\s+', texts[src])
            if len(sents) >= 5:
                keep = [sents[0]] + [s for s in sents[1:-1] if rng.random() > 0.15] + [sents[-1]]
                new_text = " ".join(keep)
            else:
                words = texts[src].split()
                new_text = " ".join([w for w in words if rng.random() > 0.1])
            out_texts.append(new_text)
            out_Y.append(Y[src].copy())
    return out_texts, np.array(out_Y, dtype=np.float32)

def make_val_split(texts, Y, val_frac=0.15, seed=42):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr, va = next(msss.split(np.zeros(len(texts)), Y))
    except Exception as e:
        print(f"  iterstrat failed ({e}), using random split")
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(texts))
        n_val = max(1, int(len(texts) * val_frac))
        va, tr = idx[:n_val], idx[n_val:]
    return ([texts[i] for i in tr], Y[tr],
            [texts[i] for i in va], Y[va])


def load_splits(event, train_path, test_path, augment=True):
    labels = get_labels(event)
    train_df, test_df = load_raw(event, train_path, test_path)
    print(f"[load] raw train={len(train_df)}  test={len(test_df)}")

    print("[load] binarizing train:")
    tr_texts, tr_Y = binarize(train_df, labels)
    print("[load] binarizing test:")
    te_texts, te_Y = binarize(test_df, labels)

    print(f"[load] train positives per class: {tr_Y.sum(axis=0).astype(int).tolist()}")
    print(f"[load] test  positives per class: {te_Y.sum(axis=0).astype(int).tolist()}")
    print(f"[load] train rows with 0 labels: {int((tr_Y.sum(1) == 0).sum())}")
    print(f"[load] test  rows with 0 labels: {int((te_Y.sum(1) == 0).sum())}")

    t_tr, y_tr, t_va, y_va = make_val_split(tr_texts, tr_Y)
    print(f"[load] after split: train={len(t_tr)}  val={len(t_va)}")

    if augment:
        t_tr, y_tr = augment_rare(t_tr, y_tr, min_count=25)
        print(f"[load] after augment: train={len(t_tr)}  per-class: {y_tr.sum(axis=0).astype(int).tolist()}")

    return {
        "train": (t_tr, y_tr),
        "val":   (t_va, y_va),
        "test":  (te_texts, te_Y),
        "label_ids":  [l["id"] for l in labels],
        "label_meta": labels,
    }
