import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BACKBONE = "microsoft/deberta-v3-large"


def _encode_descriptions(tokenizer, encoder, descriptions, device):
    encoder.eval(); encoder.to(device)
    enc = tokenizer(descriptions, padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        out = encoder(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled


class LabelDescHead(nn.Module):
    def __init__(self, label_descriptions, tokenizer, encoder, device):
        super().__init__()
        init = _encode_descriptions(tokenizer, encoder, label_descriptions, device)
        self.label_weights = nn.Parameter(init.clone())
        self.bias = nn.Parameter(torch.zeros(len(label_descriptions)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, article_emb):
        a = F.normalize(article_emb, dim=-1)
        w = F.normalize(self.label_weights, dim=-1)
        tau = torch.exp(self.log_tau).clamp(min=1e-3, max=5.0)
        return (a @ w.T) / tau + self.bias


class NarrativeModel(nn.Module):
    def __init__(self, label_descriptions, device):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
        self.encoder = AutoModel.from_pretrained(BACKBONE)
        hidden = self.encoder.config.hidden_size
        K = len(label_descriptions)
        self.dropout = nn.Dropout(0.2)
        self.desc_head = LabelDescHead(label_descriptions, self.tokenizer, self.encoder, device)
        self.linear_head = nn.Linear(hidden, K)
        self.gate = nn.Parameter(torch.zeros(K))  

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = self.dropout(pooled)
        logit_desc = self.desc_head(pooled)
        logit_lin = self.linear_head(pooled)
        g = torch.sigmoid(self.gate)
        return g * logit_desc + (1 - g) * logit_lin


class FocalHierarchyLoss(nn.Module):
    def __init__(self, pos_weight, same_side_mask, alpha=0.5, gamma=1.0,
                 cross_side_penalty=1.0, class_active_mask=None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("same_side", same_side_mask)
        if class_active_mask is None:
            class_active_mask = torch.ones_like(pos_weight)
        self.register_buffer("active", class_active_mask)
        self.alpha, self.gamma, self.csp = alpha, gamma, cross_side_penalty

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        focal = self.alpha * (1 - pt).pow(self.gamma) * bce
        if self.csp != 1.0:
            with torch.no_grad():
                gold = targets.bool().float()
                same_side_any = (gold @ self.same_side) > 0
                has_gold = gold.sum(dim=1, keepdim=True) > 0
                wrong_side = (~same_side_any) & (targets == 0).bool() & has_gold
                weight = torch.where(wrong_side, torch.full_like(logits, self.csp),
                                     torch.ones_like(logits))
                weight = weight * self.active.unsqueeze(0)
        else:
            weight = self.active.unsqueeze(0).expand_as(logits)
        return (focal * weight).mean()


def build_hierarchy_mask(label_meta):
    K = len(label_meta)
    sides = [l["side"] for l in label_meta]
    M = torch.zeros(K, K)
    for i in range(K):
        for j in range(K):
            M[i, j] = 1.0 if sides[i] == sides[j] else 0.0
    return M
