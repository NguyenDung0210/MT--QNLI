import torch
import torch.nn as nn
import torch.optim as optim
import time
from sacrebleu import corpus_bleu
from transformers import get_linear_schedule_with_warmup
import os
os.chdir('D:/Prjs/AIT3001/MT')

def train_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, learning_rate, device, tokenizer):
    start = time.time()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * epochs)
    
    model = model.to(device)
    
    with open("Transformers/training_log.txt", "w") as f:
        f.write("Start the training process!\n\n")
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                src_input_ids = batch['src_input_ids'].to(device)
                tgt_input_ids = batch['tgt_input_ids'].to(device)
                tgt_input_ids_input = tgt_input_ids[:, :-1]
                tgt_input_ids_label = tgt_input_ids[:, 1:]

                optimizer.zero_grad()
                
                outputs = model(src_input_ids, tgt_input_ids_input)
                outputs = outputs.view(-1, outputs.shape[-1])
                tgt_input_ids_label = tgt_input_ids_label.reshape(-1)
                
                loss = criterion(outputs, tgt_input_ids_label)
                total_train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if step % 50 == 0 and step != 0:
                    avg_train_loss = total_train_loss / (step + 1)
                                        
                    log = f"Epoch: {epoch + 1} / {epochs}   |   Step: {step:<3}    |   Train Loss: {avg_train_loss:.4f}"
                    print(log)
                    f.write(log + "\n")
                    f.flush()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            val_loss_epoch, val_bleu_epoch = evaluate_model(model, val_dataloader, device, criterion, tokenizer)

            log = f"Results after epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss_epoch:.4f}, Validation BLEU: {val_bleu_epoch:.4f}\n"
            print(log)
            f.write(log + "\n")
            f.flush()
        
        test_loss, test_bleu = evaluate_model(model, test_dataloader, device, criterion, tokenizer)
        f.write(f"Test Loss: {test_loss:.4f}, Test BLEU: {test_bleu:.4f}\n")

        end = time.time()
        total_time = end - start
        f.write(f"Total training time: {total_time/3600} hrs")

def evaluate_model(model, dataloader, device, criterion, tokenizer):
    model.eval()
    total_eval_loss = 0
    predictions = []
    true_labels = []

    for batch in dataloader:
        src_input_ids = batch['src_input_ids'].to(device)
        tgt_input_ids = batch['tgt_input_ids'].to(device)
        tgt_input_ids_input = tgt_input_ids[:, :-1]
        tgt_input_ids_label = tgt_input_ids[:, 1:]  

        with torch.no_grad():
            outputs = model(src_input_ids, tgt_input_ids_input)
            
            outputs = outputs.view(-1, outputs.shape[-1])
            tgt_input_ids_label = tgt_input_ids_label.reshape(-1)
            
            loss = criterion(outputs, tgt_input_ids_label)
            total_eval_loss += loss.item()
        
            preds = torch.argmax(outputs, dim=-1).view(src_input_ids.size(0), -1)
            pred_texts = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in preds]
            tgt_texts = [tokenizer.decode(tgt_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for tgt_id in tgt_input_ids]
    
            predictions.extend(pred_texts)
            true_labels.extend(tgt_texts)

    avg_val_loss = total_eval_loss / len(dataloader)
    bleu_score = corpus_bleu(predictions, [true_labels]).score

    return avg_val_loss, bleu_score
