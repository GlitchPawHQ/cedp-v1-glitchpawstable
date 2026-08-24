from tokenizers import Tokenizer

class CEDPTokenizer:
    def __init__(self, path="tokenizer/cedp.json"):
        self.tokenizer = Tokenizer.from_file(path)
        self.pad_id = self.tokenizer.token_to_id("<|pad|>")
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text, add_special_tokens=True):
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)
