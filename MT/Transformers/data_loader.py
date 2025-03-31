from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import csv
import logging
import sentencepiece as spm
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text, max_len, pad_id, return_tensors=True):
        tokens = self.sp.encode(text, out_type=int)[:max_len]
        attention_mask = [1] * len(tokens)
        
        # Padding
        while len(tokens) < max_len:
            tokens.append(pad_id)
            attention_mask.append(0)
        
        if return_tensors:
            tokens = torch.tensor(tokens)
            attention_mask = torch.tensor(attention_mask)
        
        return {"input_ids": tokens, "attention_mask": attention_mask}
    
    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        decoded_text = self.sp.DecodeIds(token_ids)

        # Optional post-processing
        if skip_special_tokens:
            decoded_text = decoded_text.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
        if clean_up_tokenization_spaces:
            decoded_text = " ".join(decoded_text.split())

        return decoded_text

    def vocab_size(self):
        return len(self.sp)

    def pad_id(self):
        return self.sp.pad_id()


def load_data(data_dir):
    train_path = f"{data_dir}/train.tsv"
    val_path = f"{data_dir}/val.tsv"
    test_path = f"{data_dir}/test.tsv"
    
    train_data = pd.read_csv(train_path, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, escapechar='\\')
    val_data = pd.read_csv(val_path, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, escapechar='\\')
    test_data = pd.read_csv(test_path, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, escapechar='\\')
    
    return train_data, val_data, test_data


class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = str(row['English'])
        tgt_text = str(row['Vietnamese'])

        src_encoding = self.tokenizer.encode(src_text, self.max_len, pad_id=self.tokenizer.pad_id())
        tgt_encoding = self.tokenizer.encode(tgt_text, self.max_len, pad_id=self.tokenizer.pad_id())
        
        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
            'src_input_ids': src_encoding['input_ids'],
            'src_attention_mask': src_encoding['attention_mask'],
            'tgt_input_ids': tgt_encoding['input_ids'],
            'tgt_attention_mask': tgt_encoding['attention_mask']
        }

def create_data_loader(data_dir, tokenizer, batch_size, max_len):
    train_data, val_data, test_data = load_data(data_dir)
    
    train_dataset = TranslationDataset(train_data, tokenizer, max_len)
    val_dataset = TranslationDataset(val_data, tokenizer, max_len)
    test_dataset = TranslationDataset(test_data, tokenizer, max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader