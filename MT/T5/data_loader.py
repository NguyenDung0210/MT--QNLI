from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv


class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        source = str(row['English'])
        target = str(row['Vietnamese'])

        inputs = self.tokenizer(
            f"translate English to Vietnamese: {source}",
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        targets = self.tokenizer(
            target,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        inputs["labels"] = targets["input_ids"].squeeze()
        return {key: val.squeeze() for key, val in inputs.items()}


def load_data(data_dir):
    train_path = f"{data_dir}/train.tsv"
    val_path = f"{data_dir}/val.tsv"
    test_path = f"{data_dir}/test.tsv"
    
    train_data = pd.read_csv(train_path, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, escapechar='\\')
    val_data = pd.read_csv(val_path, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, escapechar='\\')
    test_data = pd.read_csv(test_path, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, escapechar='\\')
    
    return train_data, val_data, test_data


def create_data_loader(data_dir, tokenizer, batch_size, max_len):
    train_data, val_data, test_data = load_data(data_dir)
    
    train_dataset = TranslationDataset(train_data, tokenizer, max_len)
    val_dataset = TranslationDataset(val_data, tokenizer, max_len)
    test_dataset = TranslationDataset(test_data, tokenizer, max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader
