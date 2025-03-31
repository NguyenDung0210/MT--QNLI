import random
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def predict(model, test_dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    questions = []
    sentences = []
    total_loss = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, b_attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            questions.extend(batch['question'])
            sentences.extend(batch['sentence'])

    test_loss = total_loss / len(test_dataloader)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print_samples(questions, sentences, predictions, true_labels, 10)

def print_samples(questions, sentences, predictions, true_labels, sample_count):
    indices = random.sample(range(len(questions)), sample_count)
    
    for i, idx in enumerate(indices):
        question = questions[idx]
        sentence = sentences[idx]
        actual = 'entailment' if true_labels[idx] == 1 else 'not_entailment'
        predicted = 'entailment' if predictions[idx] == 1 else 'not_entailment'

        print(f"Sample {i + 1}:")
        print(f"  Question: {question}")
        print(f"  Sentence: {sentence}")
        print(f"  Actual: {actual}")
        print(f"  Predicted: {predicted}")
        print("-" * 50)
