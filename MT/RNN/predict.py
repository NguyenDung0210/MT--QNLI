import torch
from sacrebleu import corpus_bleu
import random

def predict(model, test_dataloader, device, tokenizer):
    model.eval()
    predictions = []
    true_labels = []
    src_texts = []

    with torch.no_grad():
        for batch in test_dataloader:
            src_input_ids = batch['src_input_ids'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            src_text_batch = batch['src_text']
            
            hidden = model.init_hidden(src_input_ids.size(0), device)
            outputs, hidden = model(src_input_ids, hidden)
            preds = torch.argmax(outputs, dim=-1)

            pred_texts = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in preds]
            true_labels.extend([tokenizer.decode(tgt, skip_special_tokens=True, clean_up_tokenization_spaces=True) for tgt in tgt_input_ids])
            
            predictions.extend(pred_texts)
            src_texts.extend(src_text_batch)

    bleu = corpus_bleu(predictions, [true_labels]).score
    print(f"Test BLEU Score: {bleu:.4f}")

    print_samples(src_texts, true_labels, predictions, 20)

def print_samples(src_texts, true_labels, predictions, sample_count=20):
    indices = random.sample(range(len(predictions)), sample_count)
    
    for i, idx in enumerate(indices):
        src_text = src_texts[idx]
        actual = true_labels[idx]
        predicted = predictions[idx]

        print(f"Sample {i + 1}:")
        print(f"Input: {src_text}")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted}")
        print("-" * 50)
