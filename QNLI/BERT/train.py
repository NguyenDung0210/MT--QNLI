import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from data_loader import create_data_loader
import time

def train_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, epochs, criterion, f):
    model.train()
    total_loss = 0
    total_accuracy = 0

    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        correct = (preds == labels).sum().item()
        total_accuracy += correct

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 and step != 0:
            avg_loss = total_loss / (step + 1)
            avg_accuracy = total_accuracy / ((step + 1) * train_dataloader.batch_size)
            log = f"Epoch: {epoch + 1} / {epochs}   |   Step: {step:<3}    |   Loss: {avg_loss:.4f}   |   Accuracy: {avg_accuracy:.4f}"
            print(log)
            f.write(log + "\n")
            f.flush()

    avg_loss = total_loss / len(train_dataloader)
    avg_accuracy = total_accuracy / len(train_dataloader.dataset)
    return avg_loss, avg_accuracy

def train_model(train_file, valid_file, model_name, output_dir, num_train_epochs, learning_rate, per_device_train_batch_size, per_device_eval_batch_size, weight_decay, logging_steps):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataloader = create_data_loader(train_file, tokenizer, max_len=128, batch_size=per_device_train_batch_size)
    val_dataloader = create_data_loader(valid_file, tokenizer, max_len=128, batch_size=per_device_eval_batch_size)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_training_steps = len(train_dataloader) * num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    criterion = torch.nn.CrossEntropyLoss()

    with open("Final Prj/QNLI/BERT/training_log.txt", "w") as f:
        f.write("Start the training process!\n\n")

        start = time.time()
        for epoch in range(num_train_epochs):
            train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, num_train_epochs, criterion, f)
            val_loss, val_accuracy = evaluate_model(model, val_dataloader, device, criterion)

            log = f"Results after epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}\n"
            print(log)
            f.write(log + "\n")
            f.flush()

        end = time.time()
        total_time = (end - start) / 3600
        f.write(f"Total training time: {total_time:.2f} hrs\n")

    model.save_pretrained(output_dir)

def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct = (preds == labels).sum().item()
            total_accuracy += correct

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader.dataset)
    return avg_loss, avg_accuracy

def main():
    train_file = 'Final Prj/QNLI/train.tsv'
    valid_file = 'Final Prj/QNLI/val.tsv'
    model_name = "distilbert-base-uncased"
    output_dir = "Final Prj/QNLI/BERT"
    num_train_epochs = 3
    learning_rate = 1e-3
    per_device_train_batch_size = 16
    per_device_eval_batch_size = 16
    weight_decay = 0.01
    logging_steps = 62

    train_model(train_file, valid_file, model_name, output_dir, num_train_epochs, learning_rate, per_device_train_batch_size, per_device_eval_batch_size, weight_decay, logging_steps)

if __name__ == "__main__":
    main()
