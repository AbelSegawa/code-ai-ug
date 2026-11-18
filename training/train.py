
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import (
    CONFIG,
    BATCH_SIZE,
    MAX_ITERS,
    LEARNING_RATE,
    EVAL_INTERVAL,
    EVAL_ITERS
)

from model.tokenizer import build_vocab, encode
from model.transformer import CodeAI


def main():
    torch.manual_seed(1337)
    torch.set_num_threads(2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("CODE AI UG training started")
    print("Device:", device)

    data_path = ROOT / "training" / "data.txt"

    if not data_path.exists():
        raise FileNotFoundError("training/data.txt not found")

    text = data_path.read_text(encoding="utf-8")

    if len(text) < 1000:
        raise ValueError(
            "Training data is too small. Add more examples."
        )

    vocab = build_vocab(text)
    stoi = {char: index for index, char in enumerate(vocab)}

    encoded = torch.tensor(
        encode(text, stoi),
        dtype=torch.long
    )

    split_index = int(0.9 * len(encoded))

    train_data = encoded[:split_index]
    val_data = encoded[split_index:]

    block_size = CONFIG["block_size"]

    if len(train_data) <= block_size or len(val_data) <= block_size:
        raise ValueError("Dataset is too short for the configured block size.")

    def get_batch(source):
        indexes = torch.randint(
            len(source) - block_size,
            (BATCH_SIZE,)
        )

        x = torch.stack([
            source[i:i + block_size]
            for i in indexes
        ])

        y = torch.stack([
            source[i + 1:i + block_size + 1]
            for i in indexes
        ])

        return x.to(device), y.to(device)

    model = CodeAI(len(vocab), CONFIG).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        results = {}

        for name, source in [
            ("train", train_data),
            ("validation", val_data)
        ]:
            losses = torch.zeros(EVAL_ITERS)

            for k in range(EVAL_ITERS):
                x, y = get_batch(source)
                logits, loss = model(x, y)
                losses[k] = loss.item()

            results[name] = losses.mean().item()

        model.train()
        return results

    for step in range(MAX_ITERS + 1):

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss()

            print(
                f"Step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"validation loss {losses['validation']:.4f}"
            )

        x, y = get_batch(train_data)

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "code_ai_ug.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": CONFIG
        },
        checkpoint_path
    )

    print("Training completed.")
    print("Model saved:", checkpoint_path)


if __name__ == "__main__":
    main()
