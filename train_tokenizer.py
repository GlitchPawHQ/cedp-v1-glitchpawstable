from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

CORPUS = "data/processed/corpus.txt"
OUTPUT = Path("tokenizer/cedp.json")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()

special_tokens = [
    "<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>",
    "<|system|>", "<|user|>", "<|assistant|>"
]

trainer = BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=special_tokens,
)

tokenizer.train([CORPUS], trainer)
tokenizer.save(str(OUTPUT))

print(f"Tokenizer saved to {OUTPUT}")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
