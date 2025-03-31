import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def load_data(data_dir):
    train_path = f"{data_dir}/train.tsv"
    val_path = f"{data_dir}/val.tsv"
    test_path = f"{data_dir}/test.tsv"
    
    train_data = pd.read_csv(train_path, delimiter='\t', header=0)
    val_data = pd.read_csv(val_path, delimiter='\t', header=0)
    test_data = pd.read_csv(test_path, delimiter='\t', header=0)
    
    return train_data, val_data, test_data

class QNLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"entailment": 1, "not_entailment": 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row['question']
        sentence = row['sentence']
        label = row['label']

        encoding = self.tokenizer.encode_plus(
            question,
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'length': len(encoding['input_ids'].flatten().nonzero(as_tuple=True)[0]),
            'label': torch.tensor(self.label_map[label], dtype=torch.long),
            'question': question,
            'sentence': sentence
        }

def create_data_loader(data_dir, tokenizer, batch_size, max_len):
    train_data, val_data, test_data = load_data(data_dir)
    
    train_dataset = QNLIDataset(train_data, tokenizer, max_len)
    val_dataset = QNLIDataset(val_data, tokenizer, max_len)
    test_dataset = QNLIDataset(test_data, tokenizer, max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader