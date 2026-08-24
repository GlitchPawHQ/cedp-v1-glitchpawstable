from pathlib import Path
import json
import random

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/corpus.txt")

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

documents = []

for path in RAW_DIR.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            documents.append(text)

    elif path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = None

                if isinstance(obj, str):
                    text = obj
                elif isinstance(obj, dict):
                    if "text" in obj:
                        text = str(obj["text"])
                    elif "content" in obj:
                        text = str(obj["content"])
                    elif "messages" in obj:
                        parts = []
                        for message in obj["messages"]:
                            if isinstance(message, dict):
                                role = message.get("role", "user")
                                content = message.get("content", "")
                                parts.append(f"<|{role}|>\n{content}")
                        text = "\n".join(parts)

                if text and text.strip():
                    documents.append(text.strip())

random.shuffle(documents)

with OUT_FILE.open("w", encoding="utf-8") as f:
    for document in documents:
        f.write(document)
        f.write("\n\n")

print(f"Documents: {len(documents)}")
print(f"Saved corpus: {OUT_FILE}")
print(f"Characters: {OUT_FILE.stat().st_size}")
