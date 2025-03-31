import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, learning_rate, device):
    start = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    model = model.to(device)
    
    with open("Final Prj/QNLI/Transformers/training_log.txt", "w") as f:
        f.write("Start the training process!\n\n")
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            total_train_accuracy = 0
            
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch['input_ids'].to(device)
                b_attention_mask = batch['attention_mask'].to(device)
                b_labels = batch['label'].to(device)

                optimizer.zero_grad()
                
                outputs = model(b_input_ids, b_attention_mask)

                loss = criterion(outputs, b_labels)
                total_train_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).flatten()
                correct = (preds == b_labels).sum().item()
                total_train_accuracy += correct

                loss.backward()
                optimizer.step()
                
                if step % 5 == 0 and step != 0:
                    avg_train_loss = total_train_loss / (step + 1)
                    avg_train_accuracy = total_train_accuracy / ((step + 1) * train_dataloader.batch_size)
                    
                    val_loss, val_accuracy = evaluate_model(model, val_dataloader, device, criterion)
                    
                    log = f"Epoch: {epoch + 1} / {epochs}   |   Step: {step:<3}    |   Train Loss: {avg_train_loss:.4f}   |   Train Acc: {avg_train_accuracy:.4f}   |   Val Loss: {val_loss:.4f}   |   Val Acc: {val_accuracy:.4f}"
                    print(log)
                    f.write(log + "\n")
                    f.flush() 
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_train_accuracy = total_train_accuracy / len(train_dataloader.dataset)
            val_loss_epoch, val_acc_epoch = evaluate_model(model, val_dataloader, device, criterion)

            log = f"Results after epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, Validation Loss: {val_loss_epoch:.4f}, Validation Acc: {val_acc_epoch:.4f}\n"
            print(log)
            f.write(log + "\n")
            f.flush()
        
        test_loss, test_accuracy = evaluate_model(model, test_dataloader, device, criterion)
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n")

        end = time.time()
        total_time = end - start
        f.write(f"Total training time: {total_time/3600} hrs")

def evaluate_model(model, val_dataloader, device, criterion):
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)

            outputs = model(b_input_ids, b_attention_mask)

            loss = criterion(outputs, b_labels)
            total_eval_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).flatten()
            correct = (preds == b_labels).sum().item()
            total_eval_accuracy += correct

    avg_val_loss = total_eval_loss / len(val_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader.dataset)
    
    return avg_val_loss, avg_val_accuracy
