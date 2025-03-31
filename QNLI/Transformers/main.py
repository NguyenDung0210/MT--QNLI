import torch
from transformers import AutoTokenizer
from data_loader import create_data_loader
from model import Transformers
from train import train_model
from predict import predict

def main():
    data_dir = r"Final Prj\QNLI"
    max_len = 128
    batch_size = 32
    epochs = 3
    learning_rate = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_dataloader, val_dataloader, test_dataloader = create_data_loader(data_dir, tokenizer, batch_size, max_len)

    src_vocab_size = tokenizer.vocab_size
    d_model = 768
    num_heads = 12
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_seq_length = max_len

    model = Transformers(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
    train_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, learning_rate, device)

    predict(model, test_dataloader, device)

if __name__ == '__main__':
    main()