from .positional_encodeing import SinusoidalPositionalEmbedding
from .tokenizer import BPETokenizer

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L


class DNATransformerCore(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 32,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEmbedding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation=F.gelu,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.pos(h)
        key_padding_mask = attn_mask == 0
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.lm_head(h)


class BPEMLMModule(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 16,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        lr: float = 5e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.core = DNATransformerCore(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.lr = lr

    def forward(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        return self.core(x, attn)

    def _step(self, batch, stage: str):
        x, attn, labels = batch
        logits = self(x, attn)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=True if stage == "train" else False,
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @torch.inference_mode()
    def fill_sequence(
        self,
        sequence: str,
        tokenizer: BPETokenizer,
        topk: int = 1,
        device: str | None = None,
    ):
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.eval()
        masked_text = sequence.upper().replace("U", "T").replace("N", tokenizer.MASK)
        ids = tokenizer.tokenize(
            masked_text, context_len=None, truncate=False, add_special=False
        )
        attn = [0 if t == tokenizer.pad_id else 1 for t in ids]
        mask_positions = [i for i, t in enumerate(ids) if t == tokenizer.mask_id]

        x_t = torch.tensor([ids], dtype=torch.long, device=dev)
        attn_t = torch.tensor([attn], dtype=torch.long, device=dev)
        logits = self(x_t, attn_t)[0]
        V = logits.size(-1)
        special_ids = {
            tokenizer.pad_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.unk_id,
            tokenizer.mask_id,
        }

        def is_dna_tok(tok: str) -> bool:
            return len(tok) > 0 and all(c in "ACGT" for c in tok)

        allowed_ids = [
            i
            for i, t in enumerate(tokenizer.id2tok)
            if (i not in special_ids) and is_dna_tok(t)
        ]

        mask = torch.full((V,), -1e9, device=logits.device)
        mask[torch.tensor(allowed_ids, device=logits.device)] = 0.0

        logits = logits + mask
        probs = F.softmax(logits, dim=-1)

        out_ids = ids[:]
        results = []
        for i in mask_positions:
            p = probs[i]
            conf, idx = torch.max(p, dim=0)
            pred_tok = tokenizer.id2tok[int(idx)]
            out_ids[i] = int(idx)
            k = min(topk, p.numel())
            vals, inds = torch.topk(p, k=k)
            alts = [(tokenizer.id2tok[int(j)], float(v)) for v, j in zip(vals, inds)]

            results.append(
                {
                    "position_token": i,
                    "chosen_token": pred_tok,
                    "confidence": float(conf),
                    "topk": alts,
                }
            )

        filled_text = tokenizer.detokenize(out_ids, skip_special=True)
        return masked_text, filled_text, results
