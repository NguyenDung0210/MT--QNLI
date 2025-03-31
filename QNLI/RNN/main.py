import torch
from transformers import AutoTokenizer
from data_loader import create_data_loader
from rnn_model import RNNModel
from train import train_model
from predict import predict


def main():
    data_dir = r"Final Prj\QNLI"
    batch_size = 16
    max_length = 128
    epochs = 1
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataloader, val_dataloader, test_dataloader = create_data_loader(data_dir, tokenizer, batch_size, max_length)

    vocab_size = tokenizer.vocab_size
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 2
    n_layers = 2
    bidirectional = True
    dropout = 0.3

    model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout).to(device)
    train_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, learning_rate, device)

    predict(model, test_dataloader, device)


if __name__ == "__main__":
    main()