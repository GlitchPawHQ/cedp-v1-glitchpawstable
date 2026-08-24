from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ModelConfig
from src.dataset import TextDataset
from src.model import CEDPModel
from src.tokenizer import CEDPTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

config = ModelConfig()
tokenizer = CEDPTokenizer("tokenizer/cedp.json")
config.vocab_size = tokenizer.vocab_size

text = Path("data/processed/corpus.txt").read_text(encoding="utf-8")
dataset = TextDataset(text, tokenizer, config.max_seq_len)

loader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True
)

model = CEDPModel(
    vocab_size=config.vocab_size,
    max_seq_len=config.max_seq_len,
    d_model=config.d_model,
    n_heads=config.n_heads,
    n_layers=config.n_layers,
    dropout=config.dropout
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, len(loader) * config.epochs)
)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

step = 0

for epoch in range(config.epochs):
    model.train()
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
    optimizer.zero_grad()

    for batch_idx, (x, y) in enumerate(progress):
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        (loss / config.gradient_accumulation_steps).backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            progress.set_postfix(loss=f"{loss.item():.4f}", step=step)

            if step % config.save_every == 0:
                path = checkpoint_dir / f"cedp_step_{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "config": config.__dict__,
                }, path)
                print(f"\nCheckpoint saved: {path}")

final_path = checkpoint_dir / "cedp-v1-glitchpawstable-final.pt"
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": config.epochs,
    "step": step,
    "config": config.__dict__,
}, final_path)

print(f"Training finished: {final_path}")
