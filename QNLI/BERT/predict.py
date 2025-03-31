import torch
from transformers import BertForSequenceClassification, BertTokenizer
from data_loader import create_data_loader
import random
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def predict(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    questions = []
    sentences = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            questions.extend(batch['question'])
            sentences.extend(batch['sentence'])

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    precision = precision_score(true_labels, predictions, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    print_samples(questions, sentences, predictions, true_labels, 20)

def print_samples(questions, sentences, predictions, true_labels, sample_count):
    indices = random.sample(range(len(questions)), min(sample_count, len(questions)))
    
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

def main():
    test_file = 'Final Prj/QNLI/test.tsv'
    model_name = "distilbert-base-uncased"
    output_dir = "Final Prj/QNLI/BERT"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_dataloader = create_data_loader(test_file, tokenizer, max_len=128, batch_size=16)

    predict(model, test_dataloader, device)

if __name__ == "__main__":
    main()
