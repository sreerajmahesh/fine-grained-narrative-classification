import argparse, json, os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support

from data import load_splits
from model import NarrativeModel, FocalHierarchyLoss, build_hierarchy_mask


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class TextDS(Dataset):
    def __init__(self, texts, Y, tokenizer, max_len=512):
        self.texts, self.Y, self.tok, self.ml = texts, Y, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.ml,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "y": torch.from_numpy(self.Y[i])}


def tune_thresholds(logits, Y, active_mask, grid=np.arange(0.05, 0.96, 0.025)):
    probs = 1 / (1 + np.exp(-logits))
    K = Y.shape[1]
    thr = np.full(K, 0.5)
    for k in range(K):
        if not active_mask[k] or Y[:, k].sum() == 0:
            thr[k] = 0.95
            continue
        best_f1, best_t = -1, 0.5
        for t in grid:
            pred = (probs[:, k] >= t).astype(int)
            tp = ((pred == 1) & (Y[:, k] == 1)).sum()
            fp = ((pred == 1) & (Y[:, k] == 0)).sum()
            fn = ((pred == 0) & (Y[:, k] == 1)).sum()
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            if f1 > best_f1: best_f1, best_t = f1, t
        thr[k] = best_t
    return thr


def eval_metrics(Y, pred, label_ids=None):
    res = {}
    for avg in ["micro", "macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(Y, pred, average=avg, zero_division=0)
        res[f"{avg}_p"], res[f"{avg}_r"], res[f"{avg}_f1"] = float(p), float(r), float(f)
    p, r, f, _ = precision_recall_fscore_support(Y, pred, average=None, zero_division=0)
    res["per_class_f1"] = {(label_ids[i] if label_ids else str(i)): float(f[i]) for i in range(len(f))}
    return res


@torch.no_grad()
def forward_all(model, dl, device):
    model.eval()
    L, Yall = [], []
    for b in dl:
        lg = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        L.append(lg.float().cpu().numpy())
        Yall.append(b["y"].numpy())
    return np.concatenate(L), np.concatenate(Yall)


def run(event, train_path, test_path, epochs=20, bs=4, lr=5e-6, accum=4,
        max_len=512, device="cuda"):
    set_seed(42)
    print("[run] loading data...", flush=True)
    splits = load_splits(event, train_path, test_path, augment=True)
    label_ids = splits["label_ids"]
    label_meta = splits["label_meta"]
    descs = [f"{l['name']}. {l['description']}" for l in label_meta]

    t_tr, y_tr = splits["train"]
    pos = y_tr.sum(0)
    neg = len(y_tr) - pos
    active = (pos > 0).astype(np.float32)
    print(f"[train] active classes: {int(active.sum())}/{len(active)}", flush=True)
    print(f"[train] pos per class: {pos.astype(int).tolist()}", flush=True)

    pos_w = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 10.0),
                         dtype=torch.float32).to(device)
    active_t = torch.tensor(active, dtype=torch.float32).to(device)
    hier = build_hierarchy_mask(label_meta).to(device)

    print("[run] building model...", flush=True)
    model = NarrativeModel(descs, device).to(device)
    print("[run] model built and on GPU", flush=True)
    tok = model.tokenizer

    tr_dl = DataLoader(TextDS(t_tr, y_tr, tok, max_len), batch_size=bs, shuffle=True)
    va_dl = DataLoader(TextDS(*splits["val"],  tok, max_len), batch_size=bs)
    te_dl = DataLoader(TextDS(*splits["test"], tok, max_len), batch_size=bs)

    loss_fn = FocalHierarchyLoss(pos_w, hier, alpha=0.5, gamma=1.0,
                                 cross_side_penalty=1.0,
                                 class_active_mask=active_t).to(device)

    head_params = [p for n, p in model.named_parameters()
                   if "desc_head" in n or "linear_head" in n or n == "gate"]
    enc_params = [p for n, p in model.named_parameters()
                  if not ("desc_head" in n or "linear_head" in n or n == "gate")]
    opt = AdamW([
        {"params": enc_params,  "lr": lr,      "weight_decay": 0.01},
        {"params": head_params, "lr": lr * 20, "weight_decay": 0.0},
    ])

    total_steps = max(1, len(tr_dl) // accum) * epochs
    warmup = int(0.15 * total_steps)
    sch = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    best_macro, best_state, best_thr = -1, None, None
    patience, bad_epochs = 6, 0

    for ep in range(epochs):
        print(f"[train] starting epoch {ep}...", flush=True)
        model.train(); opt.zero_grad()
        epoch_loss = 0.0; nb = 0
        for i, b in enumerate(tr_dl):
            with torch.amp.autocast("cuda"):
                logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
                loss = loss_fn(logits, b["y"].to(device)) / accum
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum; nb += 1
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()

        val_logits, val_y = forward_all(model, va_dl, device)
        thr = tune_thresholds(val_logits, val_y, active)
        pred = (1 / (1 + np.exp(-val_logits)) >= thr).astype(int)
        m = eval_metrics(val_y, pred, label_ids)
        print(f"[ep {ep:2d}] loss={epoch_loss/max(nb,1):.3f}  "
              f"val micro={m['micro_f1']:.3f}  macro={m['macro_f1']:.3f}  "
              f"weighted={m['weighted_f1']:.3f}", flush=True)

        if m["macro_f1"] > best_macro:
            best_macro = m["macro_f1"]; best_thr = thr.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[train] early stop at epoch {ep}", flush=True); break

    assert best_state is not None, "No checkpoint saved — training failed"
    model.load_state_dict(best_state)
    te_logits, te_y = forward_all(model, te_dl, device)
    pred = (1 / (1 + np.exp(-te_logits)) >= best_thr).astype(int)
    m = eval_metrics(te_y, pred, label_ids)

    print(f"\n[TEST {event}] micro_f1={m['micro_f1']:.3f}  "
          f"macro_f1={m['macro_f1']:.3f}  weighted_f1={m['weighted_f1']:.3f}", flush=True)
    print("[TEST] per-class F1:", flush=True)
    for lid, fv in m["per_class_f1"].items():
        print(f"  {lid}: {fv:.3f}", flush=True)

    np.save(f"test_logits_{event}.npy", te_logits)
    np.save(f"test_y_{event}.npy", te_y)
    with open(f"test_texts_{event}.json", "w") as fout:
        json.dump(splits["test"][0], fout)
    torch.save({"state": best_state, "thresholds": best_thr.tolist(),
                "label_ids": label_ids}, f"best_{event}.pt")
    with open(f"results_{event}.json", "w") as fout:
        json.dump(m, fout, indent=2)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_len", type=int, default=512)
    a = ap.parse_args()
    run(a.event, a.train_path, a.test_path,
        epochs=a.epochs, bs=a.bs, lr=a.lr, max_len=a.max_len)
