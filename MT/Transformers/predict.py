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
            tgt_input_ids_input = tgt_input_ids[:, :-1]

            outputs = model(src_input_ids, tgt_input_ids_input)
            preds = torch.argmax(outputs, dim=-1).view(src_input_ids.size(0), -1)

            pred_texts = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in preds]
            tgt_texts = [tokenizer.decode(tgt_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for tgt_id in tgt_input_ids]
            src_texts_batch = [tokenizer.decode(src_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for src_id in src_input_ids]

            predictions.extend(pred_texts)
            true_labels.extend(tgt_texts)
            src_texts.extend(src_texts_batch)

    bleu = corpus_bleu(predictions, [true_labels]).score
    print(f"Test BLEU Score: {bleu:.4f}")

    print_samples(src_texts, true_labels, predictions, sample_count=20)

def print_samples(src_texts, true_labels, predictions, sample_count=20):
    sample_count = min(sample_count, len(predictions))
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
