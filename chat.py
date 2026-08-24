import torch
from src.model import CEDPModel
from src.tokenizer import CEDPTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/cedp-v1-glitchpawstable-final.pt"

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = checkpoint["config"]

tokenizer = CEDPTokenizer("tokenizer/cedp.json")

model = CEDPModel(
    vocab_size=config["vocab_size"],
    max_seq_len=config["max_seq_len"],
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
    dropout=config["dropout"]
).to(DEVICE)

model.load_state_dict(checkpoint["model"])
model.eval()

print("CEDP-v1-GlitchPawStable")
print("Device:", DEVICE)
print("Type /exit to quit.\n")

history = ""

while True:
    user = input("You: ").strip()

    if user.lower() == "/exit":
        break
    if not user:
        continue

    prompt = history + "<|user|>\n" + user + "\n<|assistant|>\n"

    ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    output = model.generate(
        input_ids,
        max_new_tokens=150,
        temperature=0.8,
        top_k=50
    )

    text = tokenizer.decode(output[0].tolist())

    response = text.split("<|assistant|>")[-1]
    if "<|user|>" in response:
        response = response.split("<|user|>")[0]

    response = response.strip()

    print("\nCEDP:", response, "\n")

    history += (
        "<|user|>\n" + user +
        "\n<|assistant|>\n" + response + "\n"
    )

    history = history[-12000:]
