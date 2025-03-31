import torch
from data_loader import SentencePieceTokenizer, create_data_loader
from model import Transformer
from train import train_model
from predict import predict

def main():
    data_dir = r"D:\Prjs\AIT3001\MT"
    batch_size = 12
    max_len = 80
    epochs = 20
    learning_rate = 2e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = SentencePieceTokenizer(model_path=f"{data_dir}/sp_tokenizer.model")
    train_dataloader, val_dataloader, test_dataloader = create_data_loader(data_dir, tokenizer, batch_size, max_len)
    
    src_vocab_size = tokenizer.vocab_size()
    tgt_vocab_size = tokenizer.vocab_size()
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = max_len
    dropout = 0.1

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    model.to(device)
    
    print(model)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params}")
    
    train_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, learning_rate, device, tokenizer)
    
    torch.save(model.state_dict(), f"{data_dir}/Transformers/transformer_MT.pth")
    predict(model, test_dataloader, device, tokenizer)
    
if __name__ == "__main__":
    main()
