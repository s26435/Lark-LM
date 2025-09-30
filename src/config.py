from dataclasses import dataclass


@dataclass
class MLMConfig:
    seq_len: int = 512
    mask_prob: float = 0.15
    span_lambda: float = 3.0
