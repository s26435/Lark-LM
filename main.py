import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import DataLoader, random_split

from src import BPETokenizer, MLMConfig, CSVMaskedDataset, BPEMLMModule, collate

tok = BPETokenizer("tokenizer.json")
vocab_size = len(tok)
pad_id = tok.pad_id

cfg = MLMConfig(seq_len=512, mask_prob=0.15, span_lambda=3.0)
full_ds = CSVMaskedDataset("seq.csv", tokenizer=tok, cfg=cfg)

val_size = max(1, int(0.1 * len(full_ds)))
train_size = len(full_ds) - val_size
ds_train, ds_val = random_split(full_ds, [train_size, val_size])

dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate, pin_memory=True)
dl_val   = DataLoader(ds_val,   batch_size=32, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)

model = BPEMLMModule(vocab_size=vocab_size, pad_id=pad_id, d_model=256, n_heads=8, n_layers=6, dim_ff=1024, dropout=0.1, lr=5e-4)

callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True),
        EarlyStopping(monitor="val_loss", mode="min", patience=3),
    ]
trainer = L.Trainer(
        default_root_dir="runs/",
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=50,
        callbacks=callbacks,
        gradient_clip_val=0.0,
    )


trainer.fit(model, dl_train, dl_val)

masked_text, filled_text, preds = model.fill_sequence("ACGTTNNTAGCNCCT", tokenizer=tok, topk=3)
print("MASKED:", masked_text)
print("FILLED:", filled_text)
print("TOPK@mask-positions:", preds)