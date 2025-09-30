import pandas as pd
from torch.utils.data import Dataset
import torch
import random

from .utils import sample_mask_spans, collate
from .tokenizer import BPETokenizer
from .config import MLMConfig

import lightning as L
from torch.utils.data import DataLoader


class CSVMaskedDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: BPETokenizer, cfg: MLMConfig):
        self.cfg = cfg
        self.tok = tokenizer
        df = pd.read_csv(csv_path)
        if "sequence" not in df.columns:
            raise ValueError("CSV must have a column named 'sequence'")
        self.samples = []
        for seq in df["sequence"].dropna().tolist():
            s = seq.upper().replace("U", "T")
            ids_full = self.tok.tokenize(s, context_len=None, truncate=False, add_special=False)
            # chunkowanie do stałej długości
            for i in range(0, len(ids_full), self.cfg.seq_len):
                chunk = ids_full[i:i+self.cfg.seq_len]
                if len(chunk) < self.cfg.seq_len:
                    if len(chunk) < self.cfg.seq_len // 2:
                        continue
                    pad_len = self.cfg.seq_len - len(chunk)
                    chunk = chunk + [self.tok.pad_id] * pad_len
                self.samples.append(chunk)
        if not self.samples:
            raise ValueError("Brak danych po tokenizacji CSV — sprawdź plik lub seq_len")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx][:]
        attn = [0 if t == self.tok.pad_id else 1 for t in ids]
        labels = [-100] * len(ids)
        spans = sample_mask_spans(len(ids), p=self.cfg.mask_prob, lam=self.cfg.span_lambda)
        for (a, b) in spans:
            for i in range(a, b):
                if ids[i] == self.tok.pad_id:
                    continue
                labels[i] = ids[i]
                r = random.random()
                if r < 0.8:
                    ids[i] = self.tok.mask_id
                elif r < 0.9:
                    while True:
                        rnd = random.randrange(0, len(self.tok.id2tok))
                        if self.tok.id2tok[rnd] not in {self.tok.PAD, self.tok.BOS, self.tok.EOS, self.tok.UNK, self.tok.MASK}:
                            ids[i] = rnd
                            break
                else:
                    pass
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class CSVMLMDataModule(L.LightningDataModule):
    def __init__(self, csv: str, tokenizer: BPETokenizer, seq_len: int = 512, batch_size: int = 32, workers: int = 2):
        super().__init__()
        self.csv = csv
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.workers = workers
        self.tokenizer = tokenizer
        self.cfg = MLMConfig(seq_len=seq_len)

    def setup(self, stage: str | None = None):
        full = CSVMaskedDataset(self.csv, self.tokenizer, self.cfg)
        n = len(full)
        n_val = max(1, int(0.1 * n))
        self.ds_train, self.ds_val = torch.utils.data.random_split(full, [n - n_val, n_val])

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.workers, collate_fn=collate, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.workers, collate_fn=collate, pin_memory=True)
