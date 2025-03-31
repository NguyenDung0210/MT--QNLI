import torch
import torch.nn as nn
import torch.optim as optim
import time
from sacrebleu import corpus_bleu

def train_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, learning_rate, device, tokenizer):
    start = time.time()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    model = model.to(device)
    
    with open("AIT3001/MT/RNN/training_log.txt", "w") as f:
        f.write("Start the training process!\n\n")
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                model.train()
                src_input_ids = batch['src_input_ids'].to(device)
                tgt_input_ids = batch['tgt_input_ids'].to(device)

                optimizer.zero_grad()
                
                hidden = model.init_hidden(src_input_ids.size(0), device)
                outputs, hidden = model(src_input_ids, hidden)
                
                outputs = outputs[:, :-1, :].contiguous().view(-1, outputs.shape[-1])
                tgt_input_ids = tgt_input_ids[:, 1:].contiguous().view(-1)
                
                if outputs.size(0) != tgt_input_ids.size(0):
                    if outputs.size(0) > tgt_input_ids.size(0):
                        outputs = outputs[:tgt_input_ids.size(0), :]
                    else:
                        tgt_input_ids = tgt_input_ids[:outputs.size(0)]

                loss = criterion(outputs, tgt_input_ids)
                total_train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
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
            
            torch.cuda.empty_cache()
        
        test_loss, test_bleu = evaluate_model(model, test_dataloader, device, criterion, tokenizer)
        f.write(f"Test Loss: {test_loss:.4f}, Test BLEU: {test_bleu:.4f}\n")

        end = time.time()
        total_time = end - start
        f.write(f"Total training time: {total_time/3600} hrs")

def evaluate_model(model, dataloader, device, criterion, tokenizer):
    model.eval()
    total_eval_loss = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in dataloader:
            src_input_ids = batch['src_input_ids'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            
            hidden = model.init_hidden(src_input_ids.size(0), device)
            outputs, hidden = model(src_input_ids, hidden)
            
            outputs = outputs[:, :-1, :].contiguous().view(-1, outputs.shape[-1])
            tgt_input_ids = tgt_input_ids[:, 1:].contiguous().view(-1)
            
            if outputs.size(0) != tgt_input_ids.size(0):
                if outputs.size(0) > tgt_input_ids.size(0):
                    outputs = outputs[:tgt_input_ids.size(0), :]
                else:
                    tgt_input_ids = tgt_input_ids[:outputs.size(0)]

            loss = criterion(outputs, tgt_input_ids)
            total_eval_loss += loss.item()
        
            preds = torch.argmax(outputs, dim=-1).view(src_input_ids.size(0), -1)
            reference = [tokenizer.decode(tgt_id, skip_special_tokens=True) for tgt_id in tgt_input_ids.view(src_input_ids.size(0), -1)]
            hypothese = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]

            references.extend(reference)
            hypotheses.extend(hypothese)

    avg_val_loss = total_eval_loss / len(dataloader)
    bleu_score = corpus_bleu(hypotheses, [references]).score

    torch.cuda.empty_cache()
    
    return avg_val_loss, bleu_score