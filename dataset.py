"""
Multi30k Dataset Loading and Preprocessing
DA6401 Assignment 3: Neural Machine Translation

Loads Multi30k parallel German-English corpus and builds vocabularies.
"""

import spacy
from collections import Counter
from typing import List, Tuple, Dict
from datasets import load_dataset


class Vocab:
    """Simple vocabulary class with stoi and itos mappings."""
    
    def __init__(self, tokens: List[str]):
        """
        Initialize vocabulary from a list of tokens.
        
        Args:
            tokens: List of unique tokens (words)
        """
        self.stoi = {token: idx for idx, token in enumerate(tokens)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
    
    def __len__(self):
        return len(self.stoi)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get('<unk>', 0))


class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        
        Args:
            split: One of 'train', 'validation', or 'test'
        """
        self.split = split
        
        # Load spacy tokenizers
        # try:
        #     self.src_tokenizer = spacy.load('de_core_news_sm')
        # except OSError:
        #     print("Installing German spacy model...")
        #     import os
        #     os.system('python -m spacy download de_core_news_sm')
        #     self.src_tokenizer = spacy.load('de_core_news_sm')
        
        # try:
        #     self.tgt_tokenizer = spacy.load('en_core_web_sm')
        # except OSError:
        #     print("Installing English spacy model...")
        #     import os
        #     os.system('python -m spacy download en_core_web_sm')
        #     self.tgt_tokenizer = spacy.load('en_core_web_sm')

        # Lightweight tokenizers that work everywhere
        self.src_tokenizer = spacy.blank("de")
        self.tgt_tokenizer = spacy.blank("en")
        # Load Multi30k dataset from Hugging Face
        dataset = load_dataset('bentrevett/multi30k')
        
        # Get the requested split
        if split == 'train':
            self.data = dataset['train']
        elif split == 'validation':
            self.data = dataset['validation']
        elif split == 'test':
            self.data = dataset['test']
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Initialize vocabularies
        self.src_vocab = None
        self.tgt_vocab = None
        self.src_data = None
        self.tgt_data = None
        
        # Build vocabularies
        self.build_vocab()

    def tokenize_src(self, text: str) -> List[str]:
        """Tokenize German text."""
        return [token.text.lower() for token in self.src_tokenizer.tokenizer(text)]

    def tokenize_tgt(self, text: str) -> List[str]:
        """Tokenize English text."""
        return [token.text.lower() for token in self.tgt_tokenizer.tokenizer(text)]
    
    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # Special tokens
        special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
        
        # Count tokens in source (German) and target (English)
        src_counter = Counter()
        tgt_counter = Counter()
        
        # Tokenize all sentences
        src_sentences = self.data['de']
        tgt_sentences = self.data['en']
        
        for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
            src_tokens = self.tokenize_src(src_sent)
            tgt_tokens = self.tokenize_tgt(tgt_sent)
            
            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)
        
        # Build vocabularies: special tokens + most common words
        min_freq = 1  # Minimum frequency for a token to be included
        
        src_tokens = special_tokens + [
            token for token, count in src_counter.most_common()
            if count >= min_freq
        ]
        tgt_tokens = special_tokens + [
            token for token, count in tgt_counter.most_common()
            if count >= min_freq
        ]
        
        self.src_vocab = Vocab(src_tokens)
        self.tgt_vocab = Vocab(tgt_tokens)
        
        print(f"Built vocabularies:")
        print(f"  Source vocab size: {len(self.src_vocab)}")
        print(f"  Target vocab size: {len(self.tgt_vocab)}")

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        
        Returns:
            Tuple of (src_data, tgt_data) where each is a list of token index lists
        """
        src_data = []
        tgt_data = []
        
        src_sentences = self.data['de']
        tgt_sentences = self.data['en']
        
        for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
            # Tokenize
            src_tokens = self.tokenize_src(src_sent)
            tgt_tokens = self.tokenize_tgt(tgt_sent)
            
            # Convert to indices
            src_indices = [
                self.src_vocab.stoi.get(token, self.src_vocab.stoi['<unk>'])
                for token in src_tokens
            ]
            tgt_indices = [
                self.tgt_vocab.stoi.get(token, self.tgt_vocab.stoi['<unk>'])
                for token in tgt_tokens
            ]
            
            # Add SOS and EOS
            src_indices = [self.src_vocab.stoi['<sos>']] + src_indices + [self.src_vocab.stoi['<eos>']]
            tgt_indices = [self.tgt_vocab.stoi['<sos>']] + tgt_indices + [self.tgt_vocab.stoi['<eos>']]
            
            src_data.append(src_indices)
            tgt_data.append(tgt_indices)
        
        self.src_data = src_data
        self.tgt_data = tgt_data
        
        return src_data, tgt_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single example (as raw text)."""
        return {
            'src': self.data['de'][idx],
            'tgt': self.data['en'][idx]
        }


if __name__ == "__main__":
    # Test dataset loading
    print("Loading training data...")
    train_dataset = Multi30kDataset(split='train')
    
    print("\nLoading validation data...")
    val_dataset = Multi30kDataset(split='validation')
    
    print("\nLoading test data...")
    test_dataset = Multi30kDataset(split='test')
    
    print("\nProcessing training data...")
    src_data, tgt_data = train_dataset.process_data()
    
    print(f"\nTrain set size: {len(src_data)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    print(f"\nExample 1:")
    print(f"  Source: {train_dataset.data['de'][0]}")
    print(f"  Target: {train_dataset.data['en'][0]}")