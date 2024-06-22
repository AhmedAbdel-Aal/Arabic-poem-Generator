import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class ArabicPoetryDataset(Dataset):
    def __init__(self, file_path):
        self.text = self.load_and_preprocess(file_path)
        self.chars = sorted(list(set(self.text)))
            
        # Add padding token to vocabulary
        self.pad_token = '<pad>'
        self.chars.append(self.pad_token)

        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_idx = self.char_to_idx[self.pad_token]

        
        self.sequences = self.create_sequences()

    def load_and_preprocess(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return '-'.join(text.split('\n'))

    def create_sequences(self):
        sequences = []
        lines = self.text.split('-')
        for line in lines:
            if len(line) > 1:  # Ignore empty lines
                sequences.append(line)
        return sequences
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_encoded = torch.tensor([self.char_to_idx[ch] for ch in seq], dtype=torch.long)
        return seq_encoded

    def collate_fn(self, batch):
        # Sort the batch by sequence length, descending order
        batch.sort(key=lambda x: len(x), reverse=True)
        sequences = [item for item in batch]
        
        # Pad sequences using the pad_idx
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.pad_idx)
        
        # Create targets (next character prediction)
        targets = padded_sequences[:, 1:].clone()
        padded_sequences = padded_sequences[:, :-1]
        
        # Create mask
        mask = (padded_sequences != self.pad_idx).float()
        
        return padded_sequences, targets, mask