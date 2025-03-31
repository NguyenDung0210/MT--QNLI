import torch
from transformers import MarianTokenizer
from data_loader import create_data_loader
from rnn_model import RNNModel
from train import train_model
from predict import predict

def main():
    data_dir = r"D:\Prjs\AIT3001\MT"
    batch_size = 12
    max_length = 80
    epochs = 10
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-vi')
    train_dataloader, val_dataloader, test_dataloader = create_data_loader(data_dir, tokenizer, batch_size, max_length)

    vocab_size = tokenizer.vocab_size
    embedding_dim = 100
    hidden_dim = 200
    output_dim = vocab_size
    n_layers = 2
    bidirectional = True
    dropout = 0.3

    model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout).to(device)
    train_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, learning_rate, device, tokenizer)

    torch.save(model.state_dict(), f"{data_dir}/RNN/model.pth")
    predict(model, test_dataloader, device, tokenizer)

if __name__ == "__main__":
    main()
