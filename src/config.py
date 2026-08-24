from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 16000
    max_seq_len: int = 512
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.1
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    epochs: int = 5
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 500
    save_every: int = 1000
