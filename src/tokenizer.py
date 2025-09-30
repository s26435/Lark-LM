import json
from typing import List, Dict, Optional

class BPETokenizer:
    # kanoniczne nazwy
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    MASK = "<mask>"

    # akceptowane aliasy (wiele JSONów ma [PAD], <PAD>, PAD, itp.)
    _ALIASES: Dict[str, List[str]] = {
        PAD:  ["<pad>", "[PAD]", "<PAD>", "PAD"],
        BOS:  ["<bos>", "[BOS]", "<BOS>", "BOS"],
        EOS:  ["<eos>", "[EOS]", "<EOS>", "EOS"],
        UNK:  ["<unk>", "[UNK]", "<UNK>", "UNK"],
        MASK: ["<mask>", "[MASK]", "<MASK>", "MASK"],
    }

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab_raw = data.get("dict") or data.get("vocab") or data.get("tokens")

        if isinstance(vocab_raw, list):
            self.id2tok: List[str] = list(vocab_raw)
        elif isinstance(vocab_raw, dict):
            max_id = -1
            try:
                max_id = max(int(v) for v in vocab_raw.values())
                size = max_id + 1
                self.id2tok = ["<unk>"] * size
                for tok, idx in vocab_raw.items():
                    self.id2tok[int(idx)] = tok
            except Exception:
                pairs = sorted(vocab_raw.items(), key=lambda x: (x[1], x[0]))
                self.id2tok = [tok for tok, _ in pairs]
        else:
            raise ValueError("Tokenizer JSON must contain 'dict' (list) or {'token': id} mapping.")

        self.tok2id: Dict[str, int] = {t: i for i, t in enumerate(self.id2tok)}

        self._ensure_special(self.PAD)
        self._ensure_special(self.BOS)
        self._ensure_special(self.EOS)
        self._ensure_special(self.UNK)
        self._ensure_special(self.MASK)


        self.pad_id  = self.tok2id[self.PAD]
        self.bos_id  = self.tok2id[self.BOS]
        self.eos_id  = self.tok2id[self.EOS]
        self.unk_id  = self.tok2id[self.UNK]
        self.mask_id = self.tok2id[self.MASK]

    def __len__(self) -> int:
        return len(self.id2tok)


    def _find_alias_id(self, canonical: str) -> Optional[int]:
        for name in self._ALIASES.get(canonical, [canonical]):
            if name in self.tok2id:
                return self.tok2id[name]
        return None

    def _ensure_special(self, canonical: str) -> None:
        existing = self._find_alias_id(canonical)
        if existing is not None:
            self.tok2id[canonical] = existing
            return
        new_id = len(self.id2tok)
        self.id2tok.append(canonical)
        self.tok2id[canonical] = new_id

    def tokenize(self, text: str, context_len: int | None = None, truncate: bool = True,
                 add_special: bool = False) -> List[int]:
        ids: List[int] = []
        if add_special:
            ids.append(self.bos_id)

        i, n = 0, len(text)
        while i < n:
            match_found = False
            for j in range(n, i, -1):
                sub = text[i:j]
                tid = self.tok2id.get(sub)
                if tid is not None:
                    ids.append(tid)
                    i = j
                    match_found = True
                    break
            if not match_found:
                ids.append(self.unk_id)
                i += 1

        if add_special:
            ids.append(self.eos_id)

        if context_len is not None:
            if len(ids) < context_len:
                ids.extend([self.pad_id] * (context_len - len(ids)))
            elif len(ids) > context_len:
                if truncate:
                    kept = max(1, context_len - (1 if add_special else 0))
                    ids = ids[:kept]
                    if add_special:
                        if len(ids) == context_len:
                            ids[-1] = self.eos_id
                        else:
                            ids.append(self.eos_id)
                    if len(ids) < context_len:
                        ids.extend([self.pad_id] * (context_len - len(ids)))
                else:
                    raise ValueError(f"Seq len {len(ids)} > context_len={context_len}")
        return ids

    def detokenize(self, ids: List[int], skip_special: bool = True) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        toks = [self.id2tok[i] for i in ids if not (skip_special and i in specials)]
        return "".join(toks)

    @property
    def PAD(self) -> str:  # type: ignore[override]
        return self.__class__.PAD
    @property
    def BOS(self) -> str:  # type: ignore[override]
        return self.__class__.BOS
    @property
    def EOS(self) -> str:  # type: ignore[override]
        return self.__class__.EOS
    @property
    def UNK(self) -> str:  # type: ignore[override]
        return self.__class__.UNK
    @property
    def MASK(self) -> str:  # type: ignore[override]
        return self.__class__.MASK
