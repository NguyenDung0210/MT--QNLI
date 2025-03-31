from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_loader import create_data_loader
import torch
import time
import os
os.chdir('D:/Prjs/AIT3001')


def fine_tune(model, tokenizer, train_dataloader, val_dataloader, log_file, epochs, prev_epoch, learning_rate, device):
    start = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    with open(log_file, "w") as f:
        f.write("Start the training process!\n\n")
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                if (step + 1) % 50 == 0:
                    avg_train_loss = total_loss / (step + 1)
                    log = f"Epoch: {prev_epoch + epoch + 1} / {prev_epoch + epochs}   |   Step: {step + 1:<3}    |   Train Loss: {avg_train_loss:.4f}"
                    print(log)
                    f.write(log + "\n")
                    f.flush()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()

            avg_train_loss = total_loss / len(train_dataloader)
            avg_val_loss = val_loss / len(val_dataloader)
            log = f"Results after epoch {prev_epoch + epoch + 1} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            print(log)
            f.write(log + "\n")
            f.flush()

        end = time.time()
        total_time = end - start
        f.write(f"Total fine-tuning time: {total_time/3600} hrs")
        
    model.save_pretrained("t5-en-vi")
    tokenizer.save_pretrained("t5-en-vi")
    print("Fine-tuning complete! Model saved as 't5-en-vi'.")


def show_samples(model, tokenizer, dataloader, device, num_samples=10, dataset_type="Training"):
    model.eval()
    print(f"\nResult with {dataset_type} data:")

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_samples:
                break
            inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            targets = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            outputs = model.generate(batch["input_ids"].to(device))
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i, (tgt, pred) in enumerate(zip(inputs, targets, predictions)):
                print(f"Sample {idx * len(inputs) + i + 1}:")
                print(f"Actual: {tgt}")
                print(f"Predicted: {pred}\n")

    
def main():
    data_dir = "MT"
    log_file = "MT/T5/training_log.txt"
    model_name = "MT/T5/t5-en-vi" if os.path.exists("MT/T5/t5-en-vi") else "t5-base"
    max_len = 128
    epochs = 1
    prev_epoch = 1
    batch_size = 8
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    print(model)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params}")
    
    train_dataloader, val_dataloader, test_dataloader = create_data_loader(data_dir, tokenizer, batch_size, max_len)

    fine_tune(model, tokenizer, train_dataloader, val_dataloader, log_file, epochs, prev_epoch, learning_rate, device)
    
    show_samples(model, tokenizer, train_dataloader, device, num_samples=10, dataset_type="Training")
    show_samples(model, tokenizer, test_dataloader, device, num_samples=10, dataset_type="Testing")
    
    
if __name__ == "__main__":
    main()