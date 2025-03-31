import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class QNLIDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.dataframe = pd.read_csv(file_path, delimiter='\t')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"entailment": 1, "not_entailment": 0}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        question = row['question']
        sentence = row['sentence']
        label = row['label']

        encoding = self.tokenizer(
            question,
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long),
            'question': question,
            'sentence': sentence
        }

def create_data_loader(file_path, tokenizer, max_len, batch_size):
    dataset = QNLIDataset(file_path, tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
