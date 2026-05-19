
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy
from collections import Counter
from typing import List, Tuple, Dict


UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


class Vocabulary:
    """Simple word-level vocabulary with stoi / itos mappings."""

    def __init__(self, freq_threshold: int = 2):
        self.freq_threshold = freq_threshold
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.itos: Dict[int, str] = {i: tok for tok, i in self.stoi.items()}

    def build(self, token_lists: List[List[str]]):
        counter = Counter(tok for toks in token_lists for tok in toks)
        for word, freq in counter.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx]  = word

    def __len__(self):
        return len(self.stoi)

    def numericalize(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, '<unk>')


class Multi30kDataset(Dataset):
    """
    Wraps the bentrevett/multi30k HuggingFace dataset.
    Tokenises with spaCy, builds shared Vocabulary objects,
    and returns (src_ids, tgt_ids) tensors.
    """

    # Class-level vocab so all splits share the same mapping
    src_vocab: 'Vocabulary' = None
    tgt_vocab: 'Vocabulary' = None

    def __init__(self, split: str = 'train', freq_threshold: int = 2):
        self.split = split
        self.freq_threshold = freq_threshold

        # Load HF dataset
        raw = load_dataset('bentrevett/multi30k', trust_remote_code=True)
        self.raw_data = raw[split]

        # Load spaCy models
        self.de_nlp = spacy.load('de_core_news_sm')
        self.en_nlp = spacy.load('en_core_web_sm')

        # Build vocabularies only once (on training split)
        if Multi30kDataset.src_vocab is None:
            self._build_vocabs(raw)

        # Tokenise & numericalize this split
        self.data = self._process()

    def tokenize_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.de_nlp.tokenizer(text)]

    def tokenize_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.en_nlp.tokenizer(text)]

    def _build_vocabs(self, raw_dataset):
        train_split = raw_dataset['train']
        de_tokens = [self.tokenize_de(ex['de']) for ex in train_split]
        en_tokens = [self.tokenize_en(ex['en']) for ex in train_split]

        src_vocab = Vocabulary(self.freq_threshold)
        src_vocab.build(de_tokens)
        tgt_vocab = Vocabulary(self.freq_threshold)
        tgt_vocab.build(en_tokens)

        Multi30kDataset.src_vocab = src_vocab
        Multi30kDataset.tgt_vocab = tgt_vocab
        print(f'Built vocabularies:\n  Source vocab size: {len(src_vocab)}\n  Target vocab size: {len(tgt_vocab)}')

    def _process(self):
        data = []
        for ex in self.raw_data:
            de_tok = self.tokenize_de(ex['de'])
            en_tok = self.tokenize_en(ex['en'])
            src_ids = [SOS_IDX] + Multi30kDataset.src_vocab.numericalize(de_tok) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + Multi30kDataset.tgt_vocab.numericalize(en_tok) + [EOS_IDX]
            data.append((torch.tensor(src_ids, dtype=torch.long),
                         torch.tensor(tgt_ids, dtype=torch.long)))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Pad src and tgt sequences within a batch."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def get_dataloaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = Multi30kDataset('train')
    val_ds   = Multi30kDataset('validation')
    test_ds  = Multi30kDataset('test')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=1,          shuffle=False, collate_fn=collate_fn, num_workers=2)
    return train_dl, val_dl, test_dl

