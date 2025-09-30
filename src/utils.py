import random
from typing import List, Tuple
import torch

def sample_mask_spans(L: int, p: float, lam: float) -> List[Tuple[int, int]]:
    targets = max(1, int(round(p * L)))
    spans = []
    covered = 0
    used = set()
    while covered < targets:
        span = max(1, int(random.expovariate(1.0 / lam)))
        start = random.randrange(0, L)
        end = min(L, start + span)
        if any(i in used for i in range(start, end)):
            continue
        for i in range(start, end):
            used.add(i)
            spans.append((start, end))
            covered += (end - start)
    return spans

def collate(batch):
    xs, attns, labels = zip(*batch)
    return torch.stack(xs), torch.stack(attns), torch.stack(labels)