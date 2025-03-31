from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def translate_sentence(model, tokenizer, sentence, device):
    model.eval()
    inputs = tokenizer(
        f"translate English to Vietnamese: {sentence}",
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

    translated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return translated_sentence


def main():
    # Load fine-tuned model and tokenizer
    model_dir = r"AIT3001\MT\T5\t5-en-vi"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Vietnamese Translation Inference")
    print("Type 'exit' to quit.\n")

    while True:
        sentence = input("Enter an English sentence: ")
        if sentence.lower() == "exit":
            break
        translation = translate_sentence(model, tokenizer, sentence, device)
        print(f"Vietnamese Translation: {translation}\n")


if __name__ == "__main__":
    main()
