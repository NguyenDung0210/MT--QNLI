import sentencepiece as spm

# Train SentencePiece tokenizer
spm.SentencePieceTrainer.train(
    input='dataset.txt',
    model_prefix='sp_tokenizer',
    vocab_size=8000,
    character_coverage=1.0,
    model_type='word',
    pad_id=3,
    unk_id=0,
    bos_id=1,
    eos_id=2
)

print("Tokenizer trained successfully!")