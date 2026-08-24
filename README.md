# CEDP-v1-GlitchPawStable

A real, trainable decoder-only Transformer language model implemented in PyTorch.

## Features

- Custom BPE tokenizer
- Decoder-only Transformer
- Next-token prediction training
- Checkpoint saving/loading
- GPU support when CUDA is available
- Interactive chat inference
- TXT and JSONL dataset preparation

## Requirements

- Python 3.10+
- PyTorch
- `tokenizers`
- Optional NVIDIA GPU + CUDA-enabled PyTorch for faster training

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cedp-v1-glitchpawstable.git
cd cedp-v1-glitchpawstable
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPUs, install the appropriate CUDA-enabled PyTorch build from the official PyTorch installation instructions before training if needed.

### 4. Add training data

Put `.txt` or `.jsonl` files inside:

```text
data/raw/
```

Supported JSONL examples:

```json
{"text":"Hello, I am CEDP."}
```

or conversational data:

```json
{"messages":[{"role":"user","content":"Hello!"},{"role":"assistant","content":"Hi! How can I help?"}]}
```

Only train on data you have permission to use.

### 5. Prepare the dataset

```bash
python prepare_data.py
```

This creates:

```text
data/processed/corpus.txt
```

### 6. Train the tokenizer

```bash
python train_tokenizer.py
```

This creates:

```text
tokenizer/cedp.json
```

### 7. Train CEDP

```bash
python train.py
```

Checkpoints are written to:

```text
checkpoints/
```

The final model is:

```text
checkpoints/cedp-v1-glitchpawstable-final.pt
```

### 8. Chat with the trained model

```bash
python chat.py
```

Use:

```text
/exit
```

to quit.

## Model Configuration

Edit `src/config.py` to change:

- `d_model`
- `n_heads`
- `n_layers`
- `max_seq_len`
- `batch_size`
- `learning_rate`
- `epochs`
- `gradient_accumulation_steps`

Larger models require substantially more RAM/VRAM and training time.

## Resume Training

The training script currently saves checkpoints containing:

- model weights
- optimizer state
- scheduler state
- epoch
- training step
- configuration

A checkpoint can be loaded with:

```python
checkpoint = torch.load(
    "checkpoints/cedp_step_1000.pt",
    map_location="cpu"
)
```

To implement automatic resume, modify `train.py` to load the checkpoint before entering the training loop.

## GitHub

Create an empty repository on GitHub named:

```text
cedp-v1-glitchpawstable
```

Then from this project directory:

```bash
git init
git add .
git commit -m "Initial CEDP-v1-GlitchPawStable model"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cedp-v1-glitchpawstable.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Important

The generated repository does not contain pretrained model weights or a trained tokenizer. Train them locally using the supplied scripts.

The model is a research/experimental implementation. Its quality depends heavily on the quantity, quality, formatting, and diversity of the training data.
